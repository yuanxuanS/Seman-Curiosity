from src.policy_rl.envs.utils.fmm_planner import FMMPlanner
import numpy as np
import math
from src.policy_rl.envs.utils import pose as pu
from src.policy_rl.baseline_frontier import Frontier
import torch
import cv2
import skimage
from src.policy_rl.utils.geometry_utils import rho_theta
from src.policy_rl.utils.obs_transforms import image_resize

from vsqf_utils import get_visibility_mask, get_visibility_mask_reverse, nms, neighborhoods
from src.policy_rl.utils.pointnav_policy import WrappedPointNavResNetPolicy

class vsqf_heuristic:
    def __init__(self, args, num_scenes, device):
        self.planners = [None for _ in range(num_scenes)]
        self.args = args
        self.num_scenes = num_scenes
        self.device = device
        
        self.frontier_policy = Frontier(args)
        self.frontier_policy.reset(num_scenes)
         
        self.collision_map = None
        self.last_actions = None
        self.curr_loc = None
        self.last_loc = None
        self.col_width = None
        self.visited = None
        
        # vsqf map
        self.visited_goal = None
        self.mask_sigma = np.ones(self.num_scenes) * 1.5
        self.replan = None
        self.cand_goals_map = [None]*5
         
        self.rotation_counts = None
        
        # pointnav
        pointnav_policy_path = "/home/wpp/Seman-Curiosity/data/pointnav_w"
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._last_goal = np.zeros(2)
        self._depth_image_shape = (224, 224)
        self._pointnav_stop_radius= 0.3
        self._called_stop = False
        
    def reset(self):
        self.planners = [None for _ in range(self.num_scenes)]
        
        self.frontier_policy = Frontier(self.args)
        self.frontier_policy.reset(self.num_scenes)
        
        # Episode initializations
        self.map_shape = map_shape = (self.args.map_size_cm // self.args.map_resolution,
                     self.args.map_size_cm // self.args.map_resolution)
        self.collision_map = np.zeros((self.num_scenes, map_shape[0], map_shape[1]))
        self.last_loc = [
                        [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0, 0.]
                        for _ in range(self.num_scenes)]
        self.curr_loc = [
                        [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0, 0.]
                        for _ in range(self.num_scenes)]
        self.last_actions = [None]*self.num_scenes
        self.col_width = [1]*self.num_scenes
        self.visited = np.zeros((self.num_scenes, map_shape[0], map_shape[1]))

        self.visited_goal = np.zeros((self.num_scenes, map_shape[0], map_shape[1])).astype(bool)
        self.mask_sigma = np.ones(self.num_scenes) * 1.5
        self.replan = [True for _ in range(self.num_scenes)]
        self.arrive_goal = [False for _ in range(self.num_scenes)]
        self.vis_masks = np.ones((self.num_scenes, map_shape[0], map_shape[1]))
        
        self.vsqf_goals = [[0, 0] for _ in range(self.num_scenes)]
        self.cand_goals_map = [torch.zeros((map_shape[0], map_shape[1])) for _ in range(self.num_scenes)]
        self.rotation_counts = [0]*self.num_scenes
        self.sample_num = [0]*self.num_scenes     
        
        # pointnav
        self._last_goal = np.zeros(2)  
        self._pointnav_policy.reset()

    def get_random_region(self, vsqf_map, vis_inputs, update_vis_map):
        '''
        vsqf_map: env*1*w*h
        vis_inputs: env number, dict
        '''
        selem_s = skimage.morphology.disk(1)
        selem_l = skimage.morphology.disk(3)
        exp_maps = [torch.from_numpy(skimage.morphology.dilation(np.rint(v_ip['exp_pred_full']), selem_s)) for v_ip in vis_inputs]
        # exp_maps = [torch.from_numpy(np.rint(v_ip['exp_pred_full'])) for v_ip in vis_inputs]
        exp_maps = torch.stack(exp_maps)
        
        
        # 过滤obs上的噪声
        obs_maps = []
        for e, p_input in enumerate(vis_inputs):
            obs_map = skimage.morphology.dilation(np.rint(p_input['map_pred_full']), selem_s)
            connected_colli, num_coli = skimage.morphology.label(obs_map, connectivity=1, return_num=True)
            for id in range(num_coli+1):
                    region_ = (connected_colli== id).astype(bool)
                    if region_.sum() < 50:
                        # set small collision region to traversible
                        obs_map[region_] = 0
            obs_maps.append(torch.from_numpy(obs_map))
        obs_maps = torch.stack(obs_maps)             

        exp_maps = exp_maps * (1 - obs_maps.type(torch.int))
        
        
        # 连通域分割，只在agent位置连通范围
        map_masks = []
        for e, p_input in enumerate(vis_inputs):
            
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = p_input['pose_pred']
            r, c = start_y, start_x     # 转化为格子坐标
            start = [int(r * 100.0 / self.args.map_resolution),
                    int(c * 100.0 / self.args.map_resolution)]
            start = pu.threshold_poses(start, exp_maps.shape[-2:])
        
            exp_maps_ = exp_maps[e]
            exp_maps_[int(start[0]) - 5: int(start[0])+5,       # 所在位置为可行区
                      int(start[1]) - 5: int(start[1]) + 5] = 1.
            map_mask = np.ones_like(exp_maps[e])
            connected_colli2, num_coli2 = skimage.morphology.label(exp_maps_, connectivity=1, return_num=True)
            for id in range(1, num_coli2):
                region_ = (connected_colli2== id).astype(bool)
                if region_[start[0], start[1]] > 0:
                    map_mask = region_
                    break
            map_masks.append(torch.from_numpy(map_mask))
            
        map_masks = torch.stack(map_masks)
        
        # 在可行区上，且当前位置连通的可行区上选择目标
        exp_maps = exp_maps * map_masks
        
        # vsqf选定目标物体周围的可视区, 此时exp_maps为连通区域; object_maps为和初始目标物体重叠的物体

        tgt_maps = []
        for e, v_ip in enumerate(vis_inputs):
            sem_map_ = skimage.morphology.dilation(v_ip['sem_map_pred_full'] < 5, selem_l)
            connected_colli, num_coli = skimage.morphology.label(sem_map_, connectivity=1, return_num=True)
            obj_map = v_ip['object_map_full'] < 5       # TODO 可以增加指定类别通道
            
            # 语义地图上和obj_map重叠的，重叠面积最大的为目标物体
            tgt_id = None
            max_area = -1
            for id in range(1, num_coli+1):
                region_ = (connected_colli== id).astype(bool)
                intersect = region_ * obj_map
                if intersect.sum() > 0:
                    if intersect.sum() > max_area:
                        max_area = intersect.sum()
                        tgt_id = id
            tgt_map = connected_colli== tgt_id    
            tgt_maps.append(torch.from_numpy(tgt_map))
            v_ip['target_map'] = torch.from_numpy(tgt_map)
            
        tgt_maps = torch.stack(tgt_maps)
        
        # 在explore map上，且目标的可视范围内选择
        for e in range(obs_maps.shape[0]):
            # invalid_goal_last = self.vsqf_goals[e][0] == self.vsqf_goals[e][1] and self.vsqf_goals[e][0] < 5
            if (vis_inputs[e]['sample_stage'] and self.replan[e]) or (update_vis_map[e] and vis_inputs[e]['sample_stage']):         # 
                self.vis_masks[e] = get_visibility_mask_reverse(tgt_maps[e], exp_maps[e], obs_maps[e])
 
        exp_maps = exp_maps * self.vis_masks
        exp_size = exp_maps.sum(1).sum(1)
        # 如果区域小，则选5个候选点
        self.mask_sigma = np.where(exp_size > 800, 1.5, 0.8)
            
        vsqf_map_ = vsqf_map.squeeze(1).cpu() * torch.from_numpy((1 - self.visited_goal.astype(int)))   # 去掉已经到达过的goal区域
        vsqf_map_ = vsqf_map_ * exp_maps
        
        goals = [None]*self.num_scenes
        for e in range(self.num_scenes):
            valid_map = vsqf_map_[e] > 0
            row_indices, col_indices = np.where(valid_map.cpu().numpy())
            if len(row_indices) >0:
                rand_idx = np.random.randint(0, len(row_indices))
                goals[e] = [row_indices[rand_idx], col_indices[rand_idx]]
            else:
                goals[e] = [0, 0]
                
                    
                
        for i in range(self.num_scenes):
            invalid_goal_cond = self.vis_masks[i].sum() == 0 and vis_inputs[i]['sample_stage']
            if self.replan[i] or invalid_goal_cond:
                self.vsqf_goals[i] =goals[i]

            # 如果本次无效，则从候选点中选择
                
        return self.vsqf_goals
    
    def get_best_region(self, vsqf_map, vis_inputs, update_vis_map):
        '''
        vsqf_map: env*1*w*h
        vis_inputs: env number, dict
        '''
        selem_s = skimage.morphology.disk(1)
        selem_l = skimage.morphology.disk(3)
        exp_maps = [torch.from_numpy(skimage.morphology.dilation(np.rint(v_ip['exp_pred_full']), selem_s)) for v_ip in vis_inputs]
        # exp_maps = [torch.from_numpy(np.rint(v_ip['exp_pred_full'])) for v_ip in vis_inputs]
        exp_maps = torch.stack(exp_maps)
        
        
        # 过滤obs上的噪声
        obs_maps = []
        for e, p_input in enumerate(vis_inputs):
            obs_map = skimage.morphology.dilation(np.rint(p_input['map_pred_full']), selem_s)
            connected_colli, num_coli = skimage.morphology.label(obs_map, connectivity=1, return_num=True)
            for id in range(num_coli+1):
                    region_ = (connected_colli== id).astype(bool)
                    if region_.sum() < 50:
                        # set small collision region to traversible
                        obs_map[region_] = 0
            obs_maps.append(torch.from_numpy(obs_map))
        obs_maps = torch.stack(obs_maps)             

        exp_maps = exp_maps * (1 - obs_maps.type(torch.int))
        
        
        # 连通域分割，只在agent位置连通范围
        map_masks = []
        for e, p_input in enumerate(vis_inputs):
            
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = p_input['pose_pred']
            r, c = start_y, start_x     # 转化为格子坐标
            start = [int(r * 100.0 / self.args.map_resolution),
                    int(c * 100.0 / self.args.map_resolution)]
            start = pu.threshold_poses(start, exp_maps.shape[-2:])
        
            exp_maps_ = exp_maps[e]
            exp_maps_[int(start[0]) - 5: int(start[0])+5,       # 所在位置为可行区
                      int(start[1]) - 5: int(start[1]) + 5] = 1.
            map_mask = np.ones_like(exp_maps[e])
            connected_colli2, num_coli2 = skimage.morphology.label(exp_maps_, connectivity=1, return_num=True)
            for id in range(1, num_coli2):
                region_ = (connected_colli2== id).astype(bool)
                if region_[start[0], start[1]] > 0:
                    map_mask = region_
                    break
            map_masks.append(torch.from_numpy(map_mask))
            
        map_masks = torch.stack(map_masks)
        
        # 在可行区上，且当前位置连通的可行区上选择目标
        exp_maps = exp_maps * map_masks
        
        # vsqf选定目标物体周围的可视区, 此时exp_maps为连通区域; object_maps为和初始目标物体重叠的物体

        tgt_maps = []
        for e, v_ip in enumerate(vis_inputs):
            sem_map_ = skimage.morphology.dilation(v_ip['sem_map_pred_full'] < 5, selem_l)
            connected_colli, num_coli = skimage.morphology.label(sem_map_, connectivity=1, return_num=True)
            obj_map = v_ip['object_map_full'] < 5       # TODO 可以增加指定类别通道
            
            # 语义地图上和obj_map重叠的，重叠面积最大的为目标物体
            tgt_id = None
            max_area = -1
            for id in range(1, num_coli+1):
                region_ = (connected_colli== id).astype(bool)
                intersect = region_ * obj_map
                if intersect.sum() > 0:
                    if intersect.sum() > max_area:
                        max_area = intersect.sum()
                        tgt_id = id
            tgt_map = connected_colli== tgt_id    
            tgt_maps.append(torch.from_numpy(tgt_map))
            v_ip['target_map'] = torch.from_numpy(tgt_map)
            
        tgt_maps = torch.stack(tgt_maps)
        
        # 在explore map上，且目标的可视范围内选择
        for e in range(obs_maps.shape[0]):
            # invalid_goal_last = self.vsqf_goals[e][0] == self.vsqf_goals[e][1] and self.vsqf_goals[e][0] < 5
            if (vis_inputs[e]['sample_stage'] and self.replan[e]) or (update_vis_map[e] and vis_inputs[e]['sample_stage']):         # 
                self.vis_masks[e] = get_visibility_mask_reverse(tgt_maps[e], exp_maps[e], obs_maps[e])
 
        exp_maps = exp_maps * self.vis_masks
        exp_size = exp_maps.sum(1).sum(1)
        # 如果区域小，则选5个候选点
        self.mask_sigma = np.where(exp_size > 800, 1.5, 0.8)
            
        vsqf_map_ = vsqf_map.squeeze(1).cpu() * torch.from_numpy((1 - self.visited_goal.astype(int)))   # 去掉已经到达过的goal区域
        vsqf_map_ = vsqf_map_ * exp_maps
        
        goals = [None]*self.num_scenes
        for e in range(self.num_scenes):
            if exp_size[e] > 800:      # 仅选一个
                maps_flat = vsqf_map_[e].reshape(1, -1)
                max_flat_indices = torch.argmax(maps_flat, dim=1)
                row_indices, col_indices = np.unravel_index(max_flat_indices.cpu().numpy(), (vsqf_map.shape[-2], vsqf_map.shape[-1]))
        
                goals[e] = [row_indices[0], col_indices[0]]
            else:
                if vsqf_map_[e].sum() == 0:
                    maps_flat = self.cand_goals_map[e].reshape(1, -1)
                else:
                    cand_goals_map = nms(vsqf_map_[e][None, ...], 
                        max_predictions=5, 
                        sigma=(self.mask_sigma[e], self.mask_sigma[e]),
                        gaussian=True)
                    self.cand_goals_map[e] = cand_goals_map if (cand_goals_map > 0).sum() ==5 else self.cand_goals_map[e]
                    maps_flat = (vsqf_map_[e] * cand_goals_map).reshape(1, -1)
                max_flat_indices = torch.argmax(maps_flat, dim=1)
                row_indices, col_indices = np.unravel_index(max_flat_indices.cpu().numpy(), (vsqf_map.shape[-2], vsqf_map.shape[-1]))
                goals[e] = [row_indices[0], col_indices[0]]
                    
                    
                
        for i in range(self.num_scenes):
            invalid_goal_cond = self.vis_masks[i].sum() == 0 and vis_inputs[i]['sample_stage']
            if self.replan[i] or invalid_goal_cond:
                self.vsqf_goals[i] =goals[i]

            # 如果本次无效，则从候选点中选择
                
        return self.vsqf_goals
    
    
    def get_actions(self, vis_inputs):
        
        actions = [None for _ in range(self.num_scenes)]
        for e, p_input in enumerate(vis_inputs):
            
            # if self.rotation_counts[e] < 5:        
            #     actions[e] =1       # left
            #     self.rotation_counts[e] += 1
            #     continue
            if self.rotation_counts[e] < 10:        
                actions[e] = 2
                self.rotation_counts[e] += 1
                continue
            
            if p_input['sample_stage']:
                if self.arrive_goal[e] and not self.replan[e]:
                    
                    actions[e], self.replan[e] = self.rotate_to_object(p_input, e, self.vsqf_goals[e])
                    print(f"arrive and rotation with {actions[e]}")
                    if actions[e] == 3:
                        self.sample_num[e] += 1
                    if actions[e] == 3 and self.sample_num[e] == 5:
                        self.visited_goal[e] = np.zeros((self.map_shape[0], self.map_shape[1])).astype(bool)
                        self.sample_num[e] = 0
                else:   # arrive and replan, not arrive and not replan
                    
                    actions[e], new_goal, replan_whole,  get_in_goal = self.get_actions_with_vsqf(p_input, e, self.vsqf_goals[e])
                    if new_goal[0] == self.vsqf_goals[e][0] and new_goal[1] == self.vsqf_goals[e][1]:
                        pass
                    else:
                        self.vsqf_goals[e] = new_goal 
                        p_input['frontier_goal'] = new_goal
                    
                    self.frontier_policy.collision_map[e] = self.collision_map[e]
                    self.frontier_policy.curr_loc[e] = self.curr_loc[e]
                    self.frontier_policy.last_loc[e] = self.last_loc[e]
                    self.frontier_policy.col_width[e] = self.col_width[e]
                    if get_in_goal:

                        sigma=(self.mask_sigma[e], self.mask_sigma[e])
                        gaussian=True
                        mu = torch.tensor([[self.vsqf_goals[e][1], self.vsqf_goals[e][0]]]).float()
                        visited_  = neighborhoods(mu, self.map_shape[0], self.map_shape[1], sigma, gaussian=gaussian)
                        self.visited_goal[e] = visited_.squeeze(0).numpy().astype(bool) | self.visited_goal[e].astype(bool)
                        print(f"get in vsqf goal")
                        # pointnav
                        self._pointnav_policy.reset()
                        self._last_goal = np.zeros(2) 
                        
                        self.arrive_goal[e] = True
                        self.replan[e] = False
                    else:
                        if replan_whole:        # 当前点已经不可达，重新规划目标
                            self.replan[e] = True
                        else:
                            self.replan[e] = False
                        self.arrive_goal[e] = False
                    
            else:
                # reset 
                self.cand_goals_map[e] = torch.zeros((self.map_shape[0], self.map_shape[1]))
                action, goal, short_time_goal = self.frontier_policy.get_action_one_env(p_input, e)
                actions[e] = int(action)
                
                p_input["frontier_goal"] = goal
                # p_input["short_time_goal"] = short_time_goal
                    
                self.collision_map[e] = self.frontier_policy.collision_map[e]
                self.curr_loc[e] = self.frontier_policy.curr_loc[e]
                self.last_loc[e] = self.frontier_policy.last_loc[e]
                self.col_width[e] = self.frontier_policy.col_width[e]

            self.last_actions[e] = actions[e]
        return actions
    
    def rotate_to_object(self, vis_inputs, env_idx, goal):
        
        # obj loca
        target_ = vis_inputs['target_map']
        r_idxs, c_idxs = np.where(target_ > 0)
        target_r, target_c = int(r_idxs.mean()), int(c_idxs.mean())
        
        # agent loc, orientation
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            vis_inputs['pose_pred']     # x,y,o为全局，如果要用局部的，需要减去局部原点gx1, gy1
        r, c = start_y, start_x     # 转化为格子坐标
        start = [int(r * 100.0 / self.args.map_resolution),
                 int(c * 100.0 / self.args.map_resolution)]
        start = pu.threshold_poses(start, target_.shape)
        
        
        # vector, vector orientation
        angle_st_goal = math.degrees(math.atan2(target_r - start[0],
                                                target_c - start[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        
        if relative_angle > self.args.turn_angle / 2.:
            action = 2  #3  # Right
        elif relative_angle < -self.args.turn_angle / 2.:
            action = 1  #2  # Left
        else:
            return 3, True      # capture
        return action, False

    def get_actions_with_vsqf(self, vis_input, env_idx, goal):
        
        def add_boundary(mat, value=1):
            
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat
        
        selem = skimage.morphology.disk(3)
        
        sem_map = np.rint(vis_input['sem_map_pred_full']< 5)
        sem_map = skimage.morphology.dilation(sem_map.astype(bool), selem)
        exp_map = np.rint(vis_input['exp_pred_full'])
        exp_map =  skimage.morphology.dilation(exp_map, selem)
        # 过滤噪声
        obs_map = np.rint(vis_input['map_pred_full'])
        connected_colli, num_coli = skimage.morphology.label(obs_map, connectivity=1, return_num=True)
        for id in range(num_coli):
                region_ = (connected_colli== id).astype(bool)
                if region_.sum() < 50:
                    # set small collision region to traversible
                    obs_map[region_] = 0
                        
        # obs_map = skimage.morphology.dilation(obs_map, selem)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            vis_input['pose_pred']     # x,y,o为全局，如果要用局部的，需要减去局部原点gx1, gy1
        r, c = start_y, start_x     # 转化为格子坐标
        start = [int(r * 100.0 / self.args.map_resolution),
                 int(c * 100.0 / self.args.map_resolution)]
        start = pu.threshold_poses(start, obs_map.shape)
        
        # 记录agent走过的轨迹
        self.visited[env_idx, :, :][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1       

        # update loc
        self.last_loc[env_idx] = self.curr_loc[env_idx]
        self.curr_loc[env_idx] = [start_x, start_y, start_o]
        
        
        # Collision check
        if self.last_actions[env_idx] == 0:
            x1, y1, t1 = self.last_loc[env_idx]
            x2, y2, _ = self.curr_loc[env_idx]
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width[env_idx] += 2
                if self.col_width[env_idx] == 7:
                    length = 4
                    buf = 3
                    # self.been_stuck[env_idx] = True
                    # self.stuck_cnt[env_idx] += 1
                self.col_width[env_idx] = min(self.col_width[env_idx], 5)
            else:
                self.col_width[env_idx] = 1
                # self.been_stuck[env_idx] = False
                # self.stuck_cnt[env_idx] = 0
                # self.stuck_goal[env_idx] = None

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self.args.collision_threshold:  # Collision
                width = self.col_width[env_idx]
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / self.args.map_resolution), \
                            int(c * 100 / self.args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map[env_idx].shape)
                        self.collision_map[env_idx, r, c] = 1
        
        # pointnav navigation
        num_steps = vis_input['sample_step']
        masks = torch.tensor([num_steps != 1], dtype=torch.bool, device=self.device)        #   rotation 10 times
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(np.array(goal) - np.array(self._last_goal)) > 0.1:        # 和上一个目标距离大时才作为目标
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_loc = vis_input['pose_pred'][:2]   # full pose, [c,r], real
        robot_loc_map = [int(robot_loc[0] * 100.0 / self.args.map_resolution),
                        int(robot_loc[1] * 100.0 / self.args.map_resolution)]
        heading = math.radians(vis_input['pose_pred'][2])        # 地图上朝向，逆时针为正
        goal_ = np.array([goal[1], goal[0]])
        rho, theta = rho_theta(np.array(robot_loc_map),  heading,  goal_)
        rho = rho * self.args.map_resolution / 100.     # m
        rho_theta_tensor = torch.tensor([[rho, theta]], device=self.device, dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                vis_input["depth"][None, ...],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ).to(self.device),
            "pointgoal_with_gps_compass": rho_theta_tensor.to(self.device),
        }
        # self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius:
            self._called_stop = True
            print("get in vsqf goal")
            return -1, goal, True, True
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=False).cpu().numpy()[0][0] - 1
        # 0：stop,1: forward,2:left,3: right  ——> -1,0,1,2
        return action, goal, action ==-1,  action ==-1
        
        
        x1, y1, = 0, 0
        x2, y2 = exp_map.shape
        
        # 处于sample stage，仅在eplore区域导航
        # traversible = exp_map > 0
        # traversible =traversible*(obs_map[x1:x2, y1:y2] != True)
        traversible = obs_map[x1:x2, y1:y2] != True
        traversible = traversible * (1 - sem_map.astype(int))
        
        traversible[self.collision_map[env_idx][x1:x2, y1:y2] == 1] = 0    # 去掉碰撞区
        traversible[self.visited[env_idx][x1:x2, y1:y2] == 1] = 1       # agent 轨迹也是可行区
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1      # 现在agent位置的周围

        traversible = add_boundary(traversible, value=0)
        self.planners[env_idx] = FMMPlanner(traversible)
        
        # plan with vsqf
        # goal = goal
        if pu.get_l2_distance(goal[0], start[0], goal[1],start[1]) < 5:
            print("get in vsqf goal")
            return 0, goal, True, True
        
        new_goal = self.planners[env_idx].set_goal(goal, auto_improve=True)
            
            
        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan_whole, stop = self.planners[env_idx].get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > self.args.turn_angle / 2.:
            action = 2  #3  # Right
        elif relative_angle < -self.args.turn_angle / 2.:
            action = 1  #2  # Left
        else:
            action = 0  #1  # Forward
            
        #
        return action, new_goal, replan_whole, stop