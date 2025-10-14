from src.policy_rl.envs.utils.fmm_planner import FMMPlanner
import numpy as np
import math
from src.policy_rl.envs.utils import pose as pu
from src.policy_rl.baseline_frontier import Frontier

class vsqf_heuristic:
    def __init__(self, args, num_scenes):
        self.planners = [None for _ in range(num_scenes)]
        self.args = args
        self.num_scenes = num_scenes
        
        self.frontier_policy = Frontier(args)
        self.frontier_policy.reset(num_scenes)
         
    def get_actions(self, find_goal, vis_inputs):
        
        actions = [None for _ in range(self.num_scenes)]
        for e, p_input in enumerate(vis_inputs):
            if find_goal[e]:
                actions[e] = self.get_actions_with_vsqf(p_input, e)
            else:
                action, goal, short_time_goal = self.frontier_policy.get_action_one_env(p_input, e)
                actions[e] = int(action)
        return actions
    
    def get_actions_with_vsqf(self, vis_inputs, env_idx):
        
        def add_boundary(mat, value=1):
            
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat
        
        exp_map = np.rint(vis_inputs['exp_pred_full'])
        obs_map = np.rint(vis_inputs['map_pred_full'])
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            vis_inputs['pose_pred']     # x,y,o为全局，如果要用局部的，需要减去局部原点gx1, gy1
        r, c = start_y, start_x     # 转化为格子坐标
        start = [int(r * 100.0 / self.args.map_resolution),
                 int(c * 100.0 / self.args.map_resolution)]
        start = pu.threshold_poses(start, obs_map.shape)
        
        goals = vis_inputs['long_term_goal']
        
        
        x1, y1, = 0, 0
        x2, y2 = exp_map.shape
        
        traversible = obs_map[env_idx, x1:x2, y1:y2] != True
        traversible[self.collision_map[env_idx][x1:x2, y1:y2] == 1] = 0    # 去掉碰撞区
        traversible[self.visited[env_idx][x1:x2, y1:y2] == 1] = 1       # agent 轨迹也是可行区
        traversible[env_idx, int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1      # 现在agent位置的周围

        traversible = add_boundary(traversible, value=0)
        self.planners[env_idx] = FMMPlanner(traversible)
        
        self.planners[env_idx].set_goal(goals[env_idx], auto_improve=True)
            
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
        return action