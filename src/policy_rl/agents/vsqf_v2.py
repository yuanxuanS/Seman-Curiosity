from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .utils import visualization as vu
from src.constants import color_palette
from src.vqf_constants import color_palette_vsqf
import os
import torch
from ..envs.utils import pose as pu
from ..envs.habitat.vsqf_v2_env import Vsqf_v2_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from .utils.vsqf_prediction import Vsqf_pred
from .utils.orient_prediction import Orient_pred
from src.finetune.dataset_utils import save_obs
import quaternion
from .utils.detect_utils import box_iou_calc
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes, BoxMode
from src.vqf_constants import target_coco_categories_mapping, clsid_name_maps

class Vsqf_v2_Env_Agent(Vsqf_v2_Env):
    """The VSQF environment agent class. A separate Vsqf_Env_Agent class
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
        
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.goal_name = "No"
    def reset(self):
        args = self.args
        
        obs, info = super().reset()
        obs, info = self._preprocess_obs(obs, info)

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited_vis = np.zeros(map_shape)
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        
        # visualize
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend, mode=3)
        
        return obs, info
    
    def step_and_pre(self, action, inputs, wait_env=False):
        """Function responsible for taking the action and
        preprocessing observations

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W) ? 
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep
        """
        if wait_env > 0:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            self.timestep += 1
            
            return np.zeros(self.obs.shape), 0., False, self.info
        
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

        # act and step
        action = action + np.ones_like(action)   # output: 0-2, add to 1-3
        action = {'action': action}
        obs, _, done, info = super().step(action)       # 4,256,256

        
        
        # preprocess obs
        obs, info = self._preprocess_obs(obs, info) 
        self.last_action = action['action']     
        self.obs = obs
        self.info = info


        return obs, 0., done, info
    
    def update_collision_map(self, vis_input):
        shape = self.collision_map.shape[-2:]
        self.collision_map = np.zeros((shape[0], shape[1]))
        self.collision_map[vis_input['map_pred_full'] > 0.5] = 1
    
    
    def get_mask_id(self, semantic, category, mask):
        '''
        在视野中是否有指定类别的物体, 且和mask重合
        semantic值为object id; 只要id对应的object为目标类别即可；
        用于开启检测，因此不限制范围
        '''
                
        for id in np.unique(semantic):
            if id > 0:
                # 查找对应的object
                obj = self.habitat_env.sim.semantic_scene.objects[id]
                if obj.category.name() == category:
                    if semantic.shape[0] != mask.shape[0]:
                        semantic =  cv2.resize(semantic.astype(np.uint8), mask.shape)
                    semantic_mask = semantic == id
                    intersect = semantic_mask * mask
                    if intersect.sum() > 0.1*mask.sum():
                        return True, int(obj.id[1:])
                    # if num_occ_pixels > 0.1 * semantic.shape[-1]*semantic.shape[-1]:
                    
        return False, None
    
    def get_object_id(self, semantic, category,):
        '''
        在视野中是否有指定类别的物体 
        semantic值为object id; 只要id对应的object为目标类别即可；
        用于开启检测，因此不限制范围
        '''
                
        for id in np.unique(semantic):
            if id > 0:
                # 查找对应的object
                obj = self.habitat_env.sim.semantic_scene.objects[id]
                if obj.category.name() == category:
                    num_occ_pixels = np.where(semantic == id)[0].shape[0]
                    # if num_occ_pixels > 0.1 * semantic.shape[-1]*semantic.shape[-1]:
                    return True, int(obj.id[1:])
        return False, None
    
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

        return_instance = True     # return_score: use pred score as reward; 
        sem_seg_pred, obj = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg, return_instance=return_instance)

        
        # 
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.det_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)

        if return_instance:
            save_pred_ins = False
            if save_pred_ins:
                self.save_data({'bbs': {'instances': obj}})
            
        
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)
        
        # if pred objects, pred vsqf and Orient, (在depth处理之前)
        obj = self.filter_instance(obj)
        if info['sample_stage']:
            # if info['sample_step'] > 70:    # sample stage ends
            if self.found_classes[info['target_class']]['rgbs'] == 5:
            # if info['sample_num'] == 5:      # 限制采集样本数
                info['sample_stage'] = False
                info['sample_step'] = 0
                print("sample stage ends")
                self.sampled_num += 1
                self.found_classes[info['target_class']]['num'] += 1
                    
                    
            else:       # sample stage continues
                info['sample_step'] += 1
                info['find_goal'] = False
                info['rgb_obj'] = np.zeros((256, 256, 3))
                info['depth_obj'] = np.zeros((1, 256, 256)) 
            
        else:
            
            # maskrcnn检测到时开启sample stage
            if len(obj) > 0:
                if len(obj) > 1:        # 选置信度最高
                    scores = obj.scores
                    idx = obj.scores.argmax()
                else:
                    idx = 0
                mask = obj.pred_masks[idx, ...].cpu().numpy()
                
                cls_name = clsid_name_maps[int(obj.pred_classes[idx].cpu())]
                has_obj, obj_id = self.get_mask_id(info['semantic'], cls_name, mask)
                
                if cls_name in self.found_classes.keys():
                    target_cond = self.found_classes[cls_name]['num'] < 1 and has_obj
                else:
                    target_cond = False
                if target_cond:   # 之前没找到过该类物体
                    # info['found_classes'][cls_name]['num'] += 1
                    # info['found_classes'][cls_name]['obj_id'].append(obj_id)
                    
                    self.found_classes[cls_name]['obj_id'].append(obj_id)
                    
                    
                    rgb_t = cv2.resize(rgb, (256, 256))      # 256*256
                    rgb_obj = rgb_t * mask[:, :, None]
                    depth_t = cv2.resize(depth, (256, 256)) 
                    depth_obj = depth_t * mask    # 单位cm
                    info['find_goal'] = True
                    # print(f"find goal True: {cls_name}, score {obj.scores[idx].cpu().numpy()}")
                    info['rgb_obj'] = rgb_obj
                    info['depth_obj'] = depth_obj[None, ...]
                    del rgb_t
                    del depth_t
                    # 
                    info['sample_stage'] = True
                    info['sample_step'] += 1
                    info['target_class'] = cls_name
                else:       # 采集过的物体，不处理
                    info['find_goal'] = False
                    info['rgb_obj'] = np.zeros((256, 256, 3))
                    info['depth_obj'] = np.zeros((1, 256, 256))     
                    
                    info['sample_stage'] = False
                    info['sample_step'] = 0
                # info['find_cand_goal'] = False
                # info['cand_class'] = None
                # info['cand_obj_id'] = None
            else:
                # # 从候选队列选择目标
                # if len(self.info['candidates']) > 0:
                #     print("from candidate objects")
                #     candidates_dict = self.info['candidates'].pop(0)
                #     info['find_goal'] = True
                #     info['get_cand_goal'] = True
                #     info['rgb_obj'] = candidates_dict['rgb']
                #     info['depth_obj'] = candidates_dict['depth']
                #     info['sample_stage'] = True
                #     info['sample_step'] = 1
                #     info['sample_num'] = 0
                #     info['cand_class'] = candidates_dict['class']
                #     info['cand_obj_id'] = candidates_dict['obj_id']
                # else:
                info['find_goal'] = False
                # info['get_cand_goal'] = False
                info['rgb_obj'] = np.zeros((256, 256, 3))
                info['depth_obj'] = np.zeros((1, 256, 256))     
                
                info['sample_stage'] = False
                info['sample_step'] = 0
                info['sample_num'] = 0
                info['target_class'] = None
                
                # info['find_cand_goal'] = False
                # info['cand_class'] = None
                # info['cand_obj_id'] = None
            
            
        return state, info
    
    def filter_instance(self, instance):
        if len(instance) == 0:
            return instance
        
        classes = instance.pred_classes
        new_instance = Instances(
                            pred_boxes=Boxes(torch.Tensor()),
                            image_size=instance.image_size,
                            pred_classes=torch.Tensor(),
                            pred_masks=torch.Tensor(),
                            scores=torch.Tensor(),
                        )
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx in list(target_coco_categories_mapping.keys()):
                new_instance = new_instance.cat([instance[i]])
            else:
                continue
        return new_instance
    
    def _preprocess_depth(self, depth, min_d, max_d):
        '''
        将深度值还原回设定范围, 单位cm
        '''
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        # mask1 = depth == 0
        # depth[mask1] = 100.0      #  标记为极大值：无效值
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
        
        return depth
    
    def _get_sem_pred(self, rgb, use_seg=True, return_instance=False):
        if use_seg:
            semantic_pred, self.rgb_vis, obj = self.sem_pred.get_prediction(rgb, 
                                                                            return_instance=return_instance)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 6))
            self.rgb_vis = rgb[:, :, ::-1]
        if not return_instance:
            return semantic_pred
        else:
            return semantic_pred, obj
        
        
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
        no_cat_mask = sem_map == 10     # 最后一个通道，代表无object地方
        map_mask = np.rint(map_pred) == 1       # obstacle地图
        exp_mask = np.rint(exp_pred) == 1       # explore地图
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1      # agent在地图上的位置

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2     # 将explore区域赋值2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1     # obstacle区域赋值1
        
        sem_map[vis_mask] = 3       # agent区域赋值3
        
        # vsqf map: 取间隔为0.1（放缩前），每一个间隔一个颜色
        vsqf_map = inputs['vsqf_map']
        vsqf_map_full = inputs['vsqf_map_full']
        vsqf_map[m2[None, ...]] = 14
        for i in range(10):
            score_mask = (vsqf_map > i*0.1) * (vsqf_map <= (i+1)*0.1)
            vsqf_map[score_mask] = i+2      # 从 1 开始
            
            score_mask_full = (vsqf_map_full > i *0.1) * (vsqf_map_full <= (i+1)*0.1)
            vsqf_map_full[score_mask_full] = i+2
            
        
        
        # add goal
        
        
        
        ## full map
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

        vsqf_map_full[m2_full[None, ...]] = 14
        if 'frontier_goal' in inputs:
            if inputs['frontier_goal'] is not None:
                goal = inputs['frontier_goal']
                goal_r, goal_c = goal   # r,c
                goal_x = goal_r
                goal_y = goal_c
                
                size = self.visited_vis.shape[0]
                square_size = 20
                half_size = square_size // 2
                
                for i in range(goal_x - half_size, goal_x + half_size + 1):
                    j = goal_y
                    if not inputs['sample_stage']:
                        sem_map_full[i, j] = 12
                        sem_map_full[i, j-1] = 12
                        sem_map_full[i, j+1] = 12
                    else:
                        sem_map_full[i, j] = 16
                        sem_map_full[i, j-1] = 16
                        sem_map_full[i, j+1] = 16
                
                for j in range(goal_y - half_size, goal_y + half_size + 1):
                    i = goal_x
                    if not inputs['sample_stage']:
                        sem_map_full[i, j] = 12
                        sem_map_full[i-1, j] = 12
                        sem_map_full[i+1, j] = 12
                    else:
                        sem_map_full[i, j] = 16
                        sem_map_full[i-1, j] = 16
                        sem_map_full[i+1, j] = 16
                # for i in range(goal_x - half_size, goal_x + half_size + 1):
                #     for j in range(goal_y - half_size, goal_y + half_size + 1):
                #         i = min(i, size-1)
                #         j = min(j, size-1)
                #         if not inputs['sample_stage']:
                #             sem_map_full[i, j] = 12
                #         else:
                #             sem_map_full[i, j] = 16
                            
                if 'short_time_goal' in inputs:
                    st_goal = inputs['short_time_goal']
                    st_goal_r, st_goal_c = st_goal
                    st_goal_r, st_goal_c = int(st_goal_r), int(st_goal_c)
                    st_goal_x = st_goal_r
                    st_goal_y = st_goal_c

                
                        
                    square_size = 10
                    half_size = square_size // 2
                    for i in range(st_goal_x - half_size, st_goal_x + half_size + 1):
                        for j in range(st_goal_y - half_size, st_goal_y + half_size + 1):
                            i = min(i, size-1)
                            j = min(j, size-1)
                            if not inputs['sample_stage']:
                                sem_map_full[i, j] = 12
                            else:
                                sem_map_full[i, j] = 16
                            
                        

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
        
        # vsqf
        color_pal_vsqf = [int(x * 255.) for x in color_palette_vsqf]
        if mode == "local":
            vsqf_map_vis = Image.new("P", (vsqf_map.shape[-1],
                                        vsqf_map.shape[-2]))
            vsqf_map_vis.putpalette(color_pal_vsqf)
            vsqf_map_vis.putdata(vsqf_map.flatten().astype(np.uint8))
        elif mode == "full":        
            vsqf_map_vis = Image.new("P", (vsqf_map_full.shape[-1],
                                        vsqf_map_full.shape[-2]))
            vsqf_map_vis.putpalette(color_pal_vsqf)
            vsqf_map_vis.putdata(vsqf_map_full.flatten().astype(np.uint8))
        vsqf_map_vis = vsqf_map_vis.convert("RGB")
        vsqf_map_vis = np.flipud(vsqf_map_vis)
        vsqf_map_vis = vsqf_map_vis[:, :, [2, 1, 0]]
        vsqf_map_vis = cv2.resize(vsqf_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
        
        
        rgb_vis = cv2.resize(self.rgb_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        # self.vis_image[50:530, 15:655] = rgb_vis
        # self.vis_image[50:530, 670:1150] = sem_map_vis
        # self.vis_image[50:530, 1165:1645] = vsqf_map_vis
        self.vis_image[50:530, 15:495] = rgb_vis
        self.vis_image[50:530, 510:990] = sem_map_vis
        self.vis_image[50:530, 1005:1485] = vsqf_map_vis
        
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
            
            
            
        origin = (510, 50)  
        agent_arrow = vu.get_contour_points(pos, origin)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        # agent in vsqf
        origin = (1005, 50)  
        agent_arrow = vu.get_contour_points(pos, origin)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)
        
        
        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)
            
            # fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
            #     dump_dir, self.rank, self.episode_no,
            #     self.rank, self.episode_no, self.timestep)
            # cv2.imwrite(fn, self.vis_image)
            
            pass

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)
