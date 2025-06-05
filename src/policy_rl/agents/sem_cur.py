from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .utils import visualization as vu
from src.constants import color_palette
import os
import torch
from ..envs.utils import pose as pu
from ..envs.habitat.curio_env import Seman_Curio_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.finetune.dataset_utils import save_obs
import quaternion

class Sem_Cur_Env_Agent(Seman_Curio_Env):
    """The Sem_Curiosity environment agent class. A seperate Sem_Curi_Env_Agent class
    object is used for each environment thread.

    """
    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)
        
        #
        self.visited_vis = None
        self.last_loc = None
        self.curr_loc = None
        
        # collision check
        self.last_action = None
        self.col_width= None
        self.been_stuck = None
        
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.goal_name = "No"
    def reset(self):
        args = self.args

        obs, info = super().reset()
        obs, poten_cnt = self._preprocess_obs(obs)

        info['potential_num'] = poten_cnt
        
        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited_vis = np.zeros(map_shape)
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        
        # visualize
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        
        # collision
        self.col_width = 1
        self.been_stuck = False
        return obs, info
    
    def step_and_preprocess(self, action, inputs):
        """Function responsible for taking the action and
        preprocessing observations

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W) ? 
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep
        """
        # visualize 
        self.last_loc = self.curr_loc
        # Get Map prediction
        map_pred = np.rint(inputs['map_pred'])  # 四舍五入
        
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)
        
        if self.args.visualize or self.args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / self.args.map_resolution - gx1),
                          int(c * 100.0 / self.args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])
            self._visualize(inputs)

        if self.args.agent == "forward":
            # # Collision check
            if self.last_action == 0:
                x1, y1, t1 = self.last_loc
                x2, y2, _ = self.curr_loc
                # buf = 4
                # length = 2

                if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                    self.col_width += 2
                    if self.col_width == 7:
                        length =2 # 4
                        buf = 2 #3
                    self.been_stuck = True
                    # rotation
                    action += np.ones_like(action) * np.random.randint(1, 3, 1)
                    self.col_width = min(self.col_width, 5)
                else:
                    self.col_width = 1
                    self.been_stuck = False

            # dist = pu.get_l2_distance(x1, x2, y1, y2)
            # if dist < self.args.collision_threshold:  # Collision
            #     self.been_stuck = True
                # width = self.col_width
                # for i in range(length):
                #     for j in range(width):
                #         wx = x1 + 0.05 * \
                #             ((i + buf) * np.cos(np.deg2rad(t1))
                #              + (j - width // 2) * np.sin(np.deg2rad(t1)))
                #         wy = y1 + 0.05 * \
                #             ((i + buf) * np.sin(np.deg2rad(t1))
                #              - (j - width // 2) * np.cos(np.deg2rad(t1)))
                #         r, c = wy, wx
                #         r, c = int(r * 100 / self.args.map_resolution), \
                #             int(c * 100 / self.args.map_resolution)
                #         [r, c] = pu.threshold_poses([r, c],
                #                                     self.collision_map[env_idx].shape)
                #         self.collision_map[env_idx, r, c] = 1
        # act and step
        self.last_action = action       # record 0-2
        action = action + np.ones_like(action)   # output: 0-2, add to 1-3
        action = {'action': action}
        obs, _, done, info = super().step(action)       # 4,256,256
         
        
        
        # preprocess obs
        obs, poten_cnt = self._preprocess_obs(obs) 
        self.last_action = action['action']     
        self.obs = obs
        info['potential_num'] = poten_cnt
        self.info = info
        


        return obs, 0., done, info
    
    
    
    def _preprocess_obs(self, obs, use_seg=True):
        '''
            return: state (5+C) x H x W, 
                5: RGBD + potential mask
        '''
        args = self.args
        obs = obs.transpose(1, 2, 0)
        
        rgb_ = obs[:, :, :3]     # 256,256,3
        depth_ = obs[:, :, 3:4]
        
        
        if args.det_frame_height != args.env_frame_height:
            # print(f"before resize {rgb_.shape}")
            rgb = cv2.resize(rgb_, (args.det_frame_height, args.det_frame_width))   #, rgb_.shape[-1]))
            # print(f"after resize {rgb.shape}")
            depth = cv2.resize(depth_, (args.det_frame_height, args.det_frame_width))[..., None] #, 1))
        else:
            rgb = rgb_
            depth = depth_
        del rgb_
        del depth_
    
        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        # potential mask
        potential_mask, potential_cnt = self._get_potential_mask(      # w,h,1
            rgb.astype(np.uint8), 
            depth)
        
        
        ds = args.det_frame_width // args.frame_width  # Downscaling factor: 放缩到policy输入大小
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            potential_mask = potential_mask[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, potential_mask, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, potential_cnt
    
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
    
    def _get_potential_mask(self, rgb, depth):
        self.obns_vis = self.sem_pred._get_objectness_prediction(rgb)
        poten_mask, poten_cnt = self.sem_pred.get_potential_mask(depth)
        return poten_mask, poten_cnt
    
    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 6))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred
    
    def _visualize(self, inputs, mode="full"):
        
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)       # TODO, episode_no
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        sem_map = inputs['sem_map_pred']        # local map
        sem_map_full = inputs['sem_map_pred_full']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5        # 语义id，从5开始
        sem_map_full += 5        # 语义id，从5开始
        
        # local map
        no_cat_mask = sem_map == 10     # =最后一个通道，代表没有object
        map_mask = np.rint(map_pred) == 1       # obstacle地图
        exp_mask = np.rint(exp_pred) == 1       # explore地图
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1      # agent在地图上的位置

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2     # 将explore区域赋值2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1     # obstacle区域赋值1
        
        sem_map[vis_mask] = 3       # 可视化区域赋值3
        
        # add goal
        
        
        
        # full map
        map_pred_full = inputs['map_pred_full']
        exp_pred_full = inputs['exp_pred_full']
        no_cat_mask_full = sem_map_full == 10     # 最后一个通道是什么
        map_mask_full = np.rint(map_pred_full) == 1
        exp_mask_full = np.rint(exp_pred_full) == 1
        vis_mask_full = self.visited_vis == 1

        sem_map_full[no_cat_mask_full] = 0
        m1_full = np.logical_and(no_cat_mask_full, exp_mask_full)
        sem_map_full[m1_full] = 2     # 将explore区域赋值2

        m2_full = np.logical_and(no_cat_mask_full, map_mask_full)
        sem_map_full[m2_full] = 1     # obstacle区域赋值1

        sem_map_full[vis_mask_full] = 3       # agent位置区域赋值3

        if 'frontier_goal' in inputs:
            if inputs['frontier_goal'] is not None:
                goal = inputs['frontier_goal']
                goal_r, goal_c = goal   # r,c
                goal_x = goal_r
                goal_y = goal_c
                
                st_goal = inputs['short_time_goal']
                st_goal_r, st_goal_c = st_goal
                st_goal_r, st_goal_c = int(st_goal_r), int(st_goal_c)
                st_goal_x = st_goal_r
                st_goal_y = st_goal_c

                size = self.visited_vis.shape[0]
                square_size = 20
                half_size = square_size // 2
                for i in range(goal_x - half_size, goal_x + half_size + 1):
                    for j in range(goal_y - half_size, goal_y + half_size + 1):
                        i = min(i, size-1)
                        j = min(j, size-1)
                        sem_map_full[i, j] = 12
                        
                square_size = 10
                half_size = square_size // 2
                for i in range(st_goal_x - half_size, st_goal_x + half_size + 1):
                    for j in range(st_goal_y - half_size, st_goal_y + half_size + 1):
                        i = min(i, size-1)
                        j = min(j, size-1)
                        sem_map_full[i, j] = 12
                        
        # pos
        # size = map_pred_full.shape[0]
        # square_size = 10
        # r, c = start_y, start_x     # 转化为格子坐标
        # start = [int(r * 100.0 / args.map_resolution),
        #         int(c * 100.0 / args.map_resolution)]
        # # start[1] = map_pred_full.shape[0] - start[1] 
        # start = pu.threshold_poses(start, map_pred_full.shape)
        # half_size = square_size // 2
        # for i in range(start[0] - half_size, start[0] + half_size + 1):
        #     for j in range(start[1] - half_size, start[1] + half_size + 1):
        #         i = min(i, size-1)
        #         j = min(j, size-1)
        #         sem_map_full[i, j] = 17
        # 绘制语义地图
        color_pal = [int(x * 255.) for x in color_palette]
        if mode == "local":
            sem_map_vis = Image.new("P", (sem_map.shape[1],
                                        sem_map.shape[0]))
            sem_map_vis.putpalette(color_pal)
            sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        elif mode == "full":        
            sem_map_vis = Image.new("P", (sem_map_full.shape[1],
                                        sem_map_full.shape[0]))
            sem_map_vis.putpalette(color_pal)
            sem_map_vis.putdata(sem_map_full.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
        
        obns_vis = cv2.resize(self.obns_vis, (320, 480),
                                 interpolation=cv2.INTER_NEAREST)
        rgb_vis = cv2.resize(self.rgb_vis, (320, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 10:330] = obns_vis
        self.vis_image[50:530, 335:655] = rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
        # 绘制agent位置
        if mode == "local":
            pos = (
                (start_x * 100. / args.map_resolution - gy1)        # start_x是full pose, 所以减去local bdry得到local pose
                * 480 / map_pred.shape[0],
                (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
                * 480 / map_pred.shape[1],
                np.deg2rad(-start_o)
            )
        elif mode == "full":
            pos = (
                (start_x * 100. / args.map_resolution)
                * 480 / map_pred_full.shape[0],
                (map_pred_full.shape[1] - start_y * 100. / args.map_resolution)
                * 480 / map_pred_full.shape[1],
                np.deg2rad(-start_o)
            )
            
            
            
        origin = (670, 50)  
        agent_arrow = vu.get_contour_points(pos, origin)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)
            pass

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)
