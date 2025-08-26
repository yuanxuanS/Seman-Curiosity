from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .utils import visualization as vu
from src.constants import color_palette
import os
import torch
from ..envs.utils import pose as pu
from ..envs.habitat.vqf_env import Vqf_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.finetune.dataset_utils import save_obs
import quaternion
from .utils.detect_utils import box_iou_calc
from detectron2.utils.visualizer import ColorMode, Visualizer
from src.policy_rl.envs.utils.fmm_planner import FMMPlanner
import math
import skimage
from src.vqf_constants import target_coco_categories_mapping

class VQF_Agent(Vqf_Env):
    """The VQF environment agent class. A seperate Sem_Curi_Env_Agent class
    object is used for each environment thread.

    """
    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)
        
        #
        self.visited_vis = None
        self.last_loc = None
        self.curr_loc = None
        
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)
        
        # VQF policy 
        

        self.vqf_stage = False
        self.vqf_step = -1
        self.vqf_length = 80
        self.target_cls = None
        self.target_cls_step = None
        
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.goal_name = "No"
    def reset(self):
        args = self.args
        
        obs, info = super().reset()
        obs, info = self._preprocess_obs(obs, info)
        
        # 是否进入vqf阶段
        self.vqf_stage, self.target_cls = self.stage_switch()
                
        info['vqf_stage'] = self.vqf_stage
        info['target_cls'] = self.target_cls

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
        
        return obs, info
    
    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for taking the action and
        preprocessing observations

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W) ? 
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep
        """
        
        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0
        
        action = self._plan(planner_inputs)
        
        # visualize         
        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        # act and step
        assert action >= 0, "wrong low level action"
        action = {'action': action}
        obs, _, done, info = super().step(action)       # 4,256,256

    
        # preprocess obs
        obs, info = self._preprocess_obs(obs, info) 
        self.last_action = action['action']     
        self.obs = obs
        self.info = info
        
        # 是否进入vqf阶段
        self.vqf_stage, self.target_cls = self.stage_switch()
                
        info['vqf_stage'] = self.vqf_stage
        info['target_cls'] = self.target_cls
        return obs, None, done, info
    
    def stage_switch(self):
        
        if self.vqf_stage:
            if self.vqf_step > -1 and self.vqf_step < self.vqf_length:
                self.vqf_stage = True
                self.vqf_step += 1      # TODO, 不每一步都step
                # self.target_cls = self.target_cls     # 不变
            else:
                self.vqf_stage = False
                self.vqf_step = -1
                self.target_cls = None
        else:
            if self.target_cls != None:
                self.vqf_stage = True 
                self.vqf_step += 1
                self.target_cls = self.target_cls_step
            else:
                self.vqf_stage = False
        return self.vqf_stage, self.target_cls
    
    def _preprocess_obs(self, obs, info, use_seg=True):
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

        return_score, return_instance = False, True     # return_score: use pred score as reward; 
        assert not (return_score and return_instance), \
            "Cannot return both score and instance at the same time."
        sem_seg_pred, obj = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg, return_score=return_score, return_instance=return_instance)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.det_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)

        if return_score:
            pass
        elif return_instance:
            save_pred_ins = True
            if save_pred_ins:
                self.save_data({'bbs': {'instances': obj}})
            
            anytgt = False
            for j, class_idx in enumerate(obj.pred_classes.cpu().numpy()):
                if class_idx in list(target_coco_categories_mapping.keys()):
                    idx = target_coco_categories_mapping[class_idx]

                    self.target_cls_step = idx      # TODO multi prediction
                    anytgt = True
                    break
                
            if not anytgt:
                self.target_cls_step = None
                
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, info
    
    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])  # 四舍五入
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.map_resolution - gx1),
                          int(c * 100.0 / args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:       # forward时检查是否碰撞
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:     # 和上次位置不变
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision    应该往前走但是移动距离小于阈值，判定为碰撞
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
        # FMM得到一个最近的目标点
        stg, stop = self._get_stg(map_pred, start, np.copy(goal),   # map_pred是obstacle map
                                  planning_window)

        # Deterministic Local Policy： 根据和目标的距离、偏向计算怎么行动
        if stop:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action
    
    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1): # 给地图外围加上边缘
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(   # 膨胀障碍地图
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)   # FastMatching算法
        selem = skimage.morphology.disk(10)     # 生成半径为10的圆形结构
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True        # 将目标区域扩展，并确保膨胀后的区域不属于目标区域
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
    
    def obj_exist(self, instances_1, instances_2):
        """
        Check if instances from instances_1 exist in instances_2 by comparing their bounding boxes.
        
        Args:
            instances_1: First set of instances (Instances object) to check.
            instances_2: Second set of instances (Instances object) to compare against.
        
        Returns:
            A boolean array indicating whether each instance in instances_1 exists in instances_2 (IoU > 0.5).
        """
        '''
        instances_1: Instances(n), n is the number of instances
        check if every instances_1 exists in instances_2
        return [bool, bool, ...] of length n
        '''
        
        width, height = instances_1.image_size
        pot_mp = np.zeros((height, width, 1))
        v = Visualizer(pot_mp)
        
        boxes_1 = instances_1.pred_boxes
        boxes_1 = v._convert_boxes(boxes_1)
                
        boxes_2 = instances_2.pred_boxes
        boxes_2 = v._convert_boxes(boxes_2)
        iou = box_iou_calc(boxes_1, boxes_2) # 1*num_maskbox
        is_exists = iou > 0.5
        
        return is_exists
    
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
    
    def _get_sem_pred(self, rgb, use_seg=True, return_score=False, return_instance=False):
        if use_seg:
            semantic_pred, self.rgb_vis, obj = self.sem_pred.get_prediction(rgb, 
                                                                            return_score=return_score, 
                                                                            return_instance=return_instance)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 6))
            self.rgb_vis = rgb[:, :, ::-1]
        if not (return_instance or return_score):
            return semantic_pred
        else:
            return semantic_pred, obj
        
        
    def _visualize(self, inputs, mode="local"):
        
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

        goal = inputs['goal']
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
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        
        
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
        
        rgb_vis = cv2.resize(self.rgb_vis, (640, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = rgb_vis
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
