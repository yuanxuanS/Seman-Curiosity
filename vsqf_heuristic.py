from src.policy_rl.envs.utils.fmm_planner import FMMPlanner
import numpy as np
import math
from src.policy_rl.envs.utils import pose as pu
from src.policy_rl.baseline_frontier import Frontier
import torch
import cv2
import skimage

from vsqf_utils import get_visibility_mask, get_visibility_mask_reverse, nms, neighborhoods

class vsqf_heuristic:
    def __init__(self, args, num_scenes):
        self.planners = [None for _ in range(num_scenes)]
        self.args = args
        self.num_scenes = num_scenes
        
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
        self.replan = None
        
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
        self.replan = [True for _ in range(self.num_scenes)]
        self.vis_masks = np.ones((self.num_scenes, map_shape[0], map_shape[1]))
    

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
        
        
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = p_input['pose_pred']
        r, c = start_y, start_x     # 转化为格子坐标
        start = [int(r * 100.0 / self.args.map_resolution),
                int(c * 100.0 / self.args.map_resolution)]
        start = pu.threshold_poses(start, exp_maps.shape[-2:])
        
        
        # 连通域分割，只在位置连通区域选目标
        map_masks = []
        for e, p_input in enumerate(vis_inputs):
            
            exp_maps_ = exp_maps[e]
            exp_maps_[int(start[0]) - 1: int(start[0])+2,       # 所在位置为可行区
                      int(start[1]) - 1: int(start[1]) + 2] = 1.
            map_mask = np.ones_like(exp_maps[e])
            connected_colli, num_coli = skimage.morphology.label(exp_maps_, connectivity=1, return_num=True)
            for id in range(1, num_coli):
                region_ = (connected_colli== id).astype(bool)
                if region_[start[0], start[1]] > 0:
                    map_mask = region_
                    break
            map_masks.append(torch.from_numpy(map_mask))
            
        map_masks = torch.stack(map_masks)
        
        # 在可行区上，且当前位置连通的可行区上选择目标
        exp_maps = exp_maps * map_mask
        
        # vsqf选定目标物体周围的可视区, 此时exp_maps为连通区域; object_maps为和初始目标物体重叠的物体
        # sem_map = [torch.from_numpy(skimage.morphology.dilation(np.rint(v_ip['sem_map_pred_full']), selem_s)) for v_ip in vis_inputs]
        tgt_maps = []
        for v_ip in vis_inputs:
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
            
        tgt_maps = torch.stack(tgt_maps)
        
        for e in range(obs_maps.shape[0]):
            if update_vis_map[e]:
                self.vis_masks[e] = get_visibility_mask_reverse(tgt_maps[e], exp_maps[e], obs_maps[e])
 
        vsqf_map_ = vsqf_map.squeeze(1).cpu() * torch.from_numpy((1 - self.visited_goal.astype(int)))   # 去掉已经到达过的goal区域
        vsqf_map_ = vsqf_map_ * exp_maps * self.vis_masks
        
        
        # 选最大值
        # goals_map = nms(vsqf_map_, max_predictions=1)
        # goals = []
        # for i in range(exp_maps.shape[0]):
        #     goal = torch.where(goals_map[i] > 0)
        #     if len(goal[0]) > 0:
        #         goals.append([int(goal[0][i]), int(goal[1][i])])
        #     else:
        #         goals.append([0,0])
                
        maps_flat = vsqf_map_.reshape(self.num_scenes, -1)
        max_flat_indices = torch.argmax(maps_flat, dim=1)
        row_indices, col_indices = np.unravel_index(max_flat_indices.cpu().numpy(), (vsqf_map.shape[-2], vsqf_map.shape[-1]))
        
        goals = [[row_indices[i], col_indices[i]] for i in range(self.num_scenes)]
        self.vsqf_goals = [goals[i] if self.replan[i] else self.vsqf_goals[i] for i in range(self.num_scenes)]
        return goals
    
    def get_actions(self, goals, vis_inputs):
        
        actions = [None for _ in range(self.num_scenes)]
        for e, p_input in enumerate(vis_inputs):
            if p_input['sample_stage']:
                actions[e], new_goal, get_in_goal, stop = self.get_actions_with_vsqf(p_input, e, goals[e])
                if new_goal[0] == self.vsqf_goals[e][0] and new_goal[1] == self.vsqf_goals[e][1]:
                    pass
                else:
                    self.vsqf_goals[e] = new_goal 
                    p_input['frontier_goal'] = new_goal
                
                self.frontier_policy.collision_map[e] = self.collision_map[e]
                self.frontier_policy.curr_loc[e] = self.curr_loc[e]
                self.frontier_policy.last_loc[e] = self.last_loc[e]
                self.frontier_policy.col_width[e] = self.col_width[e]
                if get_in_goal or stop:
                    # self.visited_goal[e, self.vsqf_goals[e][0] - 5:self.vsqf_goals[e][0] + 5, 
                    #                   self.vsqf_goals[e][1]-5:self.vsqf_goals[e][1]+5] = 1
                    
                    sigma=(5.0,5.0)
                    gaussian=False
                    mu = torch.tensor([[self.vsqf_goals[e][1], self.vsqf_goals[e][0]]]).float()
                    visited_  = neighborhoods(mu, self.map_shape[0], self.map_shape[1], sigma, gaussian=gaussian)
                    self.visited_goal[e] = visited_.squeeze(0).numpy().astype(bool) | self.visited_goal[e].astype(bool)
                    print(f"get in vsqf goal")
                    self.replan[e] = True
                else:
                    self.replan[e] = False
            else:
                action, goal, short_time_goal = self.frontier_policy.get_action_one_env(p_input, e)
                actions[e] = int(action)
                
                p_input["frontier_goal"] = goal
                p_input["short_time_goal"] = short_time_goal
                    
                self.collision_map[e] = self.frontier_policy.collision_map[e]
                self.curr_loc[e] = self.frontier_policy.curr_loc[e]
                self.last_loc[e] = self.frontier_policy.last_loc[e]
                self.col_width[e] = self.frontier_policy.col_width[e]
            self.last_actions[e] = actions[e]
        return actions
    
    def get_actions_with_vsqf(self, vis_inputs, env_idx, goal):
        
        def add_boundary(mat, value=1):
            
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat
        
        selem = skimage.morphology.disk(3)
        
        sem_map = np.rint(vis_inputs['sem_map_pred_full']< 5)
        sem_map = skimage.morphology.dilation(sem_map.astype(bool), selem)
        exp_map = np.rint(vis_inputs['exp_pred_full'])
        exp_map =  skimage.morphology.dilation(exp_map, selem)
        # 过滤噪声
        obs_map = np.rint(vis_inputs['map_pred_full'])
        connected_colli, num_coli = skimage.morphology.label(obs_map, connectivity=1, return_num=True)
        for id in range(num_coli):
                region_ = (connected_colli== id).astype(bool)
                if region_.sum() < 50:
                    # set small collision region to traversible
                    obs_map[region_] = 0
                        
        # obs_map = skimage.morphology.dilation(obs_map, selem)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            vis_inputs['pose_pred']     # x,y,o为全局，如果要用局部的，需要减去局部原点gx1, gy1
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
        stg_x, stg_y, get_in_goal, stop = self.planners[env_idx].get_short_term_goal(state)

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
        return action, new_goal, get_in_goal, stop