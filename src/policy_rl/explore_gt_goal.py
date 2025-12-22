
import random
import numpy as np
from src.policy_rl.envs.utils import pose as pu

class gt_goal:
    def __init__(self, args, num_scenes):
        self.args = args
        self.goals = [None]*num_scenes
        self.counts = [None]*num_scenes      # replan steps
        self.replan = [None]*num_scenes
        
        self.goals_gt = [None]*num_scenes
        self.goals_gt_add = [None]*num_scenes       # debug
        self.goal_deque = [None]*num_scenes     # 该队列用于选择目标
    def set_goals(self, obj_rel_locs, debug=False):
        '''
        obj_rel_locs: list
        '''
        # debug
        # obj_rel_locs, obj_rel_locs_add = obj_rel_locs
                
        init_loc = self.args.map_size_cm / 100.0 / 2.0
        init_agent_loc = [int(init_loc * 100.0 / self.args.map_resolution),
                               int(init_loc * 100.0 / self.args.map_resolution)]
        
        for e, data in enumerate(obj_rel_locs):
            obj_rel_loc = data[0]
            # 转为地图分辨率
            obj_abs_loc = {k:[] for k in obj_rel_loc.keys()}
            for goal, obj_loc in obj_rel_loc.items():
                for loc in obj_loc:
                    dx, dy, do = loc
                    # map resolution
                    dx_, dy_ = int(dx * 100.0 / self.args.map_resolution),  int(dy * 100.0 / self.args.map_resolution)
                    obj_c = init_agent_loc[0] + dx_
                    obj_r = init_agent_loc[1] + dy_
                    obj_r, obj_c = pu.threshold_poses([obj_r, obj_c], (479, 479))
                    obj_abs_loc[goal].append([obj_r, obj_c])
                
            self.goals_gt[e] = obj_abs_loc
            self.goal_deque[e] = list(obj_abs_loc.keys())
        
        if debug:
            for e, data in enumerate(obj_rel_locs):
                obj_rel_loc = data[1]
                # 转为地图分辨率
                obj_abs_loc = {k:[] for k in obj_rel_loc.keys()}
                for goal, obj_loc in obj_rel_loc.items():
                    for loc in obj_loc:
                        dx, dy, do = loc
                        # map resolution
                        dx_, dy_ = int(dx * 100.0 / self.args.map_resolution),  int(dy * 100.0 / self.args.map_resolution)
                        obj_c = init_agent_loc[0] + dx_
                        obj_r = init_agent_loc[1] + dy_
                        obj_r, obj_c = pu.threshold_poses([obj_r, obj_c], (479, 479))
                        obj_abs_loc[goal].append([obj_r, obj_c])
                    
                self.goals_gt_add[e] = obj_abs_loc
    
    def update_goal_deque(self, res_targets):
        for e in range(len(res_targets)):
            for goal in self.goal_deque[e]:
                if goal not in res_targets[e]:
                    self.goal_deque[e].remove(goal)
            
    def update_replan(self, env_idx):
        self.replan[env_idx] = True
    def get_goals(self, p_input, env_idx):
        
        
            
            
        if self.goals[env_idx] is None:
            self.replan[env_idx] = True
            self.counts[env_idx] = 0
        else:
            # if not self.replan[env_idx]:
            # replan if get close to goal
            # update replan state
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
            if len(self.goal_deque[env_idx]) > 0:
                objs_loc = self.goals_gt[env_idx][self.goal_deque[env_idx][0]]
                objs_loc_add = self.goals_gt_add[env_idx][self.goal_deque[env_idx][0]]
            else:
                objs_loc = self.goals_gt[env_idx]["chair"]
                objs_loc_add = self.goals_gt_add[env_idx]["chair"]
            self.goal_deque[env_idx] = self.goal_deque[env_idx][1:] + self.goal_deque[env_idx][:1]        # deque更新
            idx = random.choice(range(len(objs_loc)))
            goal = objs_loc[idx]
            #debug
            goal_add = objs_loc_add[idx]
        else:
            goal = self.goals[env_idx]
            goal_add = None
        
        self.goals[env_idx] = goal
        return self.goals[env_idx], goal_add