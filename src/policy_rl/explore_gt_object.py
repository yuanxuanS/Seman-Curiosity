
import random
import numpy as np
from src.policy_rl.envs.utils import pose as pu

class gt_object:
    def __init__(self, args, num_scenes):
        self.args = args
        self.goals = [None]*num_scenes
        self.counts = [None]*num_scenes      # replan steps
        self.replan = [None]*num_scenes
        
        self.goals_gt = [None]*num_scenes
        
    def set_goals(self, obj_rel_locs, debug=False):
        '''
        obj_rel_locs: list
        '''
        # debug
        # obj_rel_locs, obj_rel_locs_add = obj_rel_locs
                
        init_loc = self.args.map_size_cm / 100.0 / 2.0
        init_agent_loc = [int(init_loc * 100.0 / self.args.map_resolution),
                               int(init_loc * 100.0 / self.args.map_resolution)]
        
        for e, obj_rel_loc in enumerate(obj_rel_locs):
            # 转为地图分辨率
            obj_abs_loc = []
            for obj_loc in obj_rel_loc:
                dx, dy, do = obj_loc
                # map resolution
                dx_, dy_ = int(dx * 100.0 / self.args.map_resolution),  int(dy * 100.0 / self.args.map_resolution)
                obj_c = init_agent_loc[0] + dx_
                obj_r = init_agent_loc[1] + dy_
                obj_r, obj_c = pu.threshold_poses([obj_r, obj_c], (479, 479))
                obj_abs_loc.append([obj_r, obj_c])
                
            self.goals_gt[e] = obj_abs_loc
        
    
            
    def update_replan(self, env_idx):
        self.replan[env_idx] = True
        
    def get_goals(self, p_input, env_idx):

        if self.goals[env_idx] is None:
            self.replan[env_idx] = True
            self.counts[env_idx] = 0
        else:
            # if not self.replan[env_idx]:
            # update replan state：replan if get close to goal
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                p_input['pose_pred']
            x2, y2, _ = start_x, start_y, start_o       #self.curr_loc[env_idx]
            r, c = y2, x2     # 转化为格子坐标
            start = [int(r * 100.0 / self.args.map_resolution),
                    int(c * 100.0 / self.args.map_resolution)]
            map_pred = np.rint(p_input['map_pred_full'])
            start = pu.threshold_poses(start, map_pred.shape)
            goal_r, goal_c = self.goals[env_idx][0], self.goals[env_idx][1]
            dis = pu.get_l2_distance(goal_c, start[1], goal_r, start[0])
            if dis < 15:     # for grid distance
                self.replan[env_idx] = True
                print(f"get in goal {self.replan[env_idx]}")
                # self.selected_goal[env_idx, goal_r-2:goal_r+2, goal_c-2:goal_c+2] = 1.
            else:
                self.replan[env_idx] = self.counts[env_idx] >= 200       # if long time
                    
            if self.replan[env_idx]:
                self.counts[env_idx] = 0 
            else:
                self.counts[env_idx] += 1
            
        # 
        if self.replan[env_idx]:
            goal = self.goals_gt[env_idx][0]
            self.goals_gt[env_idx] = self.goals_gt[env_idx][1:] + self.goals_gt[env_idx][:1]        # deque更新
        else:
            goal = self.goals[env_idx]
        
        self.goals[env_idx] = goal
        return self.goals[env_idx]