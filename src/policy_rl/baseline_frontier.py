import numpy as np
import cv2
from src.policy_rl.envs.utils import pose as pu
from src.policy_rl.envs.utils.fmm_planner import FMMPlanner
from src.policy_rl.arguments import get_args
from src.policy_rl.envs import make_vec_envs
from src.policy_rl.maps import Maps_Env
import torch
import math
import skimage
import random

class Frontier:
    '''
    frontier baseline on full map
    '''
    def __init__(self, args):
        self.frontiers = None
        self.last_actions = []
        self.args = args
        
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_actions = None
        self.last_goal = None
        
        self.visited = None
        self.visited_vis = None
        self.collision_map = None
        self.count_forward_actions = None
        
        self.goals = None
        self.short_time_goals = None
        self.replan = None
        self.counts = None      # replan steps
        self.rotation_counts = None
        
        self.been_stuck = None
        self.stuck_cnt = None
        self.stuck_goal = None
        
        # self.obs_shape = None
        
        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        
    def reset(self, env_nums):
        args = self.args
        self.env_nums = env_nums
        # self.obs_shape = obs_shape

        # Episode initializations
        self.map_shape = map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros((env_nums, map_shape[0], map_shape[1]))
        self.visited = np.zeros((env_nums, map_shape[0], map_shape[1]))
        # self.visited_vis = np.zeros(map_shape)
        
        self.col_width = [1]*env_nums
        self.count_forward_actions = [0]*env_nums
        self.curr_loc = [
                        [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
                        for _ in range(env_nums)]
        self.last_loc = [
                        [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
                        for _ in range(env_nums)]
        self.last_actions = [None]*env_nums
        self.last_goal = [[None, None]]*env_nums
        
        self.goals = [None]*env_nums
        self.short_time_goals = [None]*env_nums
        self.replan = [True]*env_nums
        self.counts = [0]*env_nums
        self.rotation_counts = [0]*env_nums
        
        self.been_stuck = [False]*env_nums
        self.stuck_cnt = [0]*env_nums
        self.stuck_goal = [None]*env_nums
    
    def get_actions(self, vis_inputs):
        
        actions = []
        for e, p_input in enumerate(vis_inputs):
            
            # update loc: real distance (m)
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                p_input['pose_pred']
            self.last_loc[e] = self.curr_loc[e]
            self.curr_loc[e] = [start_x, start_y, start_o]
            
            if self.rotation_counts[e] < 10:        
                action = 2  # left
                actions.append(action)
                self.rotation_counts[e] += 1
                continue
            
            # update replan
            x2, y2, _ = self.curr_loc[e]
            if self.goals[e] is None:
                self.replan[e] = True
                self.counts[e] = 0
            else:
                # replan if get close to goal
                goal_r, goal_c = self.goals[e][0], self.goals[e][1]
                if abs(goal_c - x2) < 10 and abs(goal_r - y2) < 10:     # for grid distance
                    self.replan[e] = True
                else:
                    self.replan[e] = False
                # replan if long time or get in goal
                self.replan[e] = self.counts[e] == 50 or self.replan[e]
                if self.replan[e]:
                    self.counts[e] = 0 
                else:
                    self.counts[e] += 1
                
            
            if self.replan[e]:  # compute goal on full map

                fmap, lagst_contrs = self.get_frontier_map(p_input, e)
                gain_fmap = self.get_frontier_gains(fmap, p_input, lagst_contrs)
                goal = self.sample_frontier(gain_fmap, e)
                self.goals[e] = goal
                # print(f"replan in env :{e}, goal {goal}")
            else:
                goal = self.goals[e]

            self.last_goal[e] = goal
            action, short_time_goal = self.get_determine_action(p_input, goal, e)
            print(f"goal: {goal}, short_time_goal: {short_time_goal}, action: {action}")
            actions.append(action)
            self.short_time_goals[e] = short_time_goal
        return np.array(actions), self.goals, self.short_time_goals
    
    def get_determine_action(self, p_input, goal, env_idx):
        '''
        object-oriented 里面的determin policy
            goal: r, c
        '''
        # convert goal to goal map            
        goal_map = np.zeros((self.map_shape[1], self.map_shape[0]))
        goal_map[goal[0], goal[1]] = 1
            
        action, short_time_goal = self._plan(p_input, goal_map, env_idx)
        return action, short_time_goal
        
    def _plan(self, planner_inputs, goal, env_idx):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred_full'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose in full map (x,y,o)
                                 and planning window of local map (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        # self.last_loc[env_idx] = self.curr_loc[env_idx]

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred_full'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']     # x,y,o为全局，如果要用局部的，需要减去局部原点gx1, gy1
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x     # 转化为格子坐标
        start = [int(r * 100.0 / args.map_resolution),
                 int(c * 100.0 / args.map_resolution)]
        start = pu.threshold_poses(start, map_pred.shape)

        # self.visited[env_idx, gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        self.visited[env_idx, :, :][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1       # 记录agent走过的轨迹


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
                    self.been_stuck[env_idx] = True
                    self.stuck_cnt[env_idx] += 1
                self.col_width[env_idx] = min(self.col_width[env_idx], 5)
            else:
                self.col_width[env_idx] = 1
                self.been_stuck[env_idx] = False
                self.stuck_cnt[env_idx] = 0

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
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
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map[env_idx].shape)
                        self.collision_map[env_idx, r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window, env_idx)

        if self.been_stuck[env_idx] and self.stuck_cnt[env_idx] >= 40:
            if self.stuck_goal[env_idx] is None:     # 卡住后，寻找新的目标点

                navigable_indices = np.argwhere(self.visited[env_idx, :, :] > 0)
                goal_ = np.array([0, 0])
                for _ in range(100):    # 随机从探索过的地图中找新的目标点goal
                    random_index = np.random.choice(len(navigable_indices))
                    goal_ = navigable_indices[random_index]
                    if pu.get_l2_distance(goal_[0], start[0], goal_[1], start[1]) > 16:   # 在lcoal map上找一个较远距离的目标点
                        goal_ = pu.threshold_poses(goal_, map_pred.shape)                
                        self.stuck_goal[env_idx] = [int(goal_[0]), int(goal_[1])]      # 在全局地图的位置
            else:
                goal_ = np.array([self.stuck_goal[env_idx][0], self.stuck_goal[env_idx][1]])
                goal_ = pu.threshold_poses(goal_, map_pred.shape)
            stg = goal_
        # Deterministic Local Policy
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
            action = 2  #3  # Right
        elif relative_angle < -self.args.turn_angle / 2.:
            action = 1  #2  # Left
        else:
            action = 0  #1  # Forward
            
        # check stuck
        self.last_actions[env_idx] = action
        return action, stg

    
    def _get_stg(self, grid, start, goal, planning_window, env_idx):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        # traversible[self.collision_map[env_idx][gx1:gx2, gy1:gy2]
        #             [x1:x2, y1:y2] == 1] = 0
        # traversible[self.visited[env_idx][gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[self.collision_map[env_idx]
                    [x1:x2, y1:y2] == 1] = 0    # 去掉碰撞区
        traversible[self.visited[env_idx][x1:x2, y1:y2] == 1] = 1       # agent 轨迹也是可行区

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1      # 现在agent位置的周围

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
    
    def sample_frontier(self, gains_fmap, env_idx):
        '''
            get a frontier goal with max gain
                gains_fmap: 1, full_w, full_h
        '''

        gfmap = gains_fmap.squeeze(0)
        # flat_indices = np.where(gfmap.flatten() > 1)[0]
        # random_index = random.choice(flat_indices)
        # max_idx = np.unravel_index(random_index, gfmap.shape)
        while True:
            max_idx = np.argmax(gfmap)
            max_idx = np.unravel_index(max_idx, gfmap.shape)    # max_idx: r, c
            if max_idx[0] == self.last_goal[env_idx][0] and max_idx[1] == self.last_goal[env_idx][1]:
                gfmap[max_idx[0], max_idx[1]] = 0. 
                continue
            else:
                break
            
        return max_idx
        
        
    def get_frontier_gains(self, frontier_map, p_input, largest_contours, type="2"):
        '''

        compute frontiers' gains.
            frontier_map: ndarray
            largest_contours: contours of frontiers
            
        return:
            gains_fmaps: ndarray: num_envs, H, W
        '''
        # Compute unexplored free-space starting from each frontier: PONI, SemanticMapPrecomputedDataset. get_masks_and_labels()
        

        # floor_map = out_semmap[0, FLOOR_ID] # (H, W)
        # unexp_map = ~torch.any(in_semmap[0], dim=0) # (H, W)
        # unexp_floor_map = floor_map & unexp_map # (H, W)
        # unexp_floor_map = unexp_floor_map.cpu().numpy()
        
        out_area_pfs = None
            
        exp_map = np.rint(p_input["exp_pred_full"])
        unk_map = 1 - exp_map
        
        # Identify connected components of unexplored floor space
        unexp_floor_map = unk_map.astype(np.uint8) * 255
        ncomps, comp_labs, _, _ = cv2.connectedComponentsWithStats(
            unexp_floor_map, 4 , cv2.CV_32S
        )
        # breakpoint()
        # Compute contours of frontiers
        # contours = None
        # contours, _ = cv2.findContours(
        #     frontier_map.astype(np.uint8),
        #     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE        # 仅检测最外层的轮廓
        # )
        # contours = [contour[:, 0].tolist() for contour in contours] # Clean format

        # for largest 5 contours
        # largest_contours = sorted(
        #     contours, key=lambda cnt: len(cnt), reverse=True
        # )[:5]
        contour_stats = [0.0 for _ in range(len(largest_contours))]
        # For each connected component, find the intersecting frontiers and
        # add area to them. (as frontier gain)
        kernel = np.ones((5, 5))
        self.grid_size = unexp_floor_map.shape
        
        
        for i in range(1, ncomps):
            comp = (comp_labs == i).astype(np.float32)
            comp_area = comp.sum().item()   # * (self.grid_size[0] *self.grid_size[1])
            # dilate
            comp = cv2.dilate(comp, kernel, iterations=1)
            # intersect with frontiers
            for j, contour in enumerate(largest_contours):
                intersection = 0.0
                for x, y in contour:
                    intersection += comp[y, x]
                if intersection > 0:
                    contour_stats[j] += comp_area
            
        # normalize_area_by_constant = False
        # if normalize_area_by_constant:
        #     total_area = self.cfg.max_unexp_area
        # else:
        #     total_area = floor_map.sum().item() * (self.grid_size ** 2) / 2.0
        
        def calculate_centroid(points):
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            centroid_x = sum(x_coords) / len(points)
            centroid_y = sum(y_coords) / len(points)
            return (centroid_x, centroid_y)
        # create frontier gain map
        gains_fmap = torch.zeros(self.grid_size, dtype=torch.float)
        for stat, contour in zip(contour_stats, largest_contours):
            score = stat
            center_c, center_r = calculate_centroid(contour)
            gains_fmap[int(center_r), int(center_c)] = score
        
        # Create out areas map
        # out_area_pfs = torch.zeros(self.grid_size, dtype=torch.float) # (H, W)
        # for stat, contour in zip(contour_stats, largest_contours):
        #     # Use linear scoring
        #     score = stat
        #     # score = np.clip(stat / (total_area + EPS), 0.0, 1.0)
        #     for x, y in contour:
        #         out_area_pfs[y, x] = score
        # Dilate the area map
        # out_area_pfs = out_area_pfs.unsqueeze(0).unsqueeze(1) # (1, 1, H, W)
        # out_area_pfs = torch.nn.functional.max_pool2d(
        #     out_area_pfs, 7, stride=1, padding=3
        # )
        # out_area_pfs = out_area_pfs.squeeze(1) # (1, H, W)

        return gains_fmap
    
    def get_frontier_maps(self, inputs_envs):
        frontier_maps = []
        for e, p_input in enumerate(inputs_envs):
            fmap = self.get_frontier_map(p_input)
            frontier_maps.append(fmap)
        return frontier_maps
            
    def get_frontier_map(self, planner_inputs, env_idx):
        """Function responsible for computing frontiers in the input map
            from IEVE-main: Instance_Exp_Env_Agent: get_frontier_map()
        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        args = self.args

        obs_map = np.rint(planner_inputs["map_pred_full"])
        exp_map = np.rint(planner_inputs["exp_pred_full"])
        # selem = skimage.morphology.disk(3)
        # obs_map = skimage.morphology.dilation(
        #     obs_map,
        #     selem)
        # exp_map = skimage.morphology.dilation(
        #     exp_map,
        #     selem)
        
        # compute free and unexplored maps
        free_map = (1 - obs_map) * exp_map
        unk_map = 1 - exp_map
        # Clean maps
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(free_map, cv2.MORPH_CLOSE, kernel)
        unk_map[free_map == 1] = 0
        unk_map_shiftup = np.pad(
            unk_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unk_map_shiftdown = np.pad(
            unk_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unk_map_shiftleft = np.pad(
            unk_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unk_map_shiftright = np.pad(
            unk_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (H, W)
        frontiers = frontiers.astype(np.uint8)
        # Select only large-enough frontiers
        contours, _ = cv2.findContours(
            frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # self.frontier_vis = np.zeros((240, 240, 3), np.uint8)
        # contours_vis, _ = cv2.findContours(
        #     frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # self.frontier_vis = cv2.drawContours(self.frontier_vis,contours_vis,-1,(122,122,255),3) 
        
        largest_contours = []
        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)  
            # Only pick largest 1 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:5]:
                contour = np.array(contour)
                # Select only the central point of the contour
                lc = len(contour)
                if lc > 0:
                    new_frontiers[contour[lc // 2, 1], contour[lc // 2, 0]] = 1
            frontiers = new_frontiers
            largest_contours = contours[:5]
        # breakpoint()
        frontiers = frontiers > 0
        # Mask out frontiers very close to the agent
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        ## Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / args.map_resolution)
        frontiers[
            (start[0] - ncells) : (start[0] + ncells + 1),
            (start[1] - ncells) : (start[1] + ncells + 1),
        ] = False
        # Handle edge case where frontier becomes zero
        if not np.any(frontiers):
            # Set a random location to True
            for _ in range(5):
                rand_y = np.random.randint(start[0] - ncells, start[0] + ncells + 1)
                rand_x = np.random.randint(start[1] - ncells, start[1] + ncells + 1)
                frontiers[rand_y, rand_x] = True

        
        return frontiers, largest_contours
        
if __name__ == "__main__":
    args = get_args()
    
    device = args.device = torch.device("cuda:1" if args.cuda else "cpu")   # 训练的gpu

    num_scenes = args.num_processes
    
    envs = make_vec_envs(args) 
    obs, infos = envs.reset()
    
    maps = Maps_Env(args)
    local_map, local_pose = maps.update_semantic_map(obs, infos)

    # for visualize
    full_map = maps.full_map
    vis_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(vis_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = maps.get_all_pose()[e]
        
        p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = maps.get_all_pose()[e]
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5       # 有物体时，为了argmax时不选最后通道
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()   # 如果无object，选最后一个通道
            full_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred_full'] = full_map[e, 4:, :, :].argmax(0).cpu().numpy()

    l_policy = Frontier(args)
    l_policy.reset(num_scenes)
    actions = l_policy.get_actions(vis_inputs)
    print(f"action: {actions}")
    