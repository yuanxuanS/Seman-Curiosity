import torch
from .model import Semantic_Mapping
import numpy as np
from src.policy_rl.arguments import get_args
from .envs import make_vec_envs

class Maps_Env:
    def __init__(self, args):
        super(Maps_Env, self).__init__()

        self.args = args

        device = self.device = args.device
        num_scenes = args.num_processes

        self.semantic_map = Semantic_Mapping(args).to(device)
        self.semantic_map.eval()

        # Calculating full and local map sizes,     
        map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = args.num_sem_categories + 4  # num channels
        print(f"seman num in mapps: {args.num_sem_categories}")
        self.num_scenes = num_scenes
        # Initializing full and local map
        self.local_map = torch.zeros(num_scenes, nc, self.local_w,
                            self.local_h).float().to(device)
        self.full_map = torch.zeros(num_scenes, nc, self.full_w, self.full_h).float().to(device)
    
        # Initial full and local pose
        self.full_pose = torch.zeros(num_scenes, 3).float().to(device)
        self.local_pose = torch.zeros(num_scenes, 3).float().to(device)
        
        # Origin of local map
        self.origins = np.zeros((num_scenes, 3))
        
        # Local Map Boundaries
        self.lmb = np.zeros((num_scenes, 4)).astype(int)

        # allpose
        self.pose_inputs = np.zeros((self.num_scenes, 7))   # pose in full map, local bdry in full map
        # initialize
        self._init_map_and_pose()

    def get_all_pose(self):
        return self.pose_inputs[:, :]


    def _init_map_and_pose(self):
        '''
        从fullmap初始化local map
        '''
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                              (self.local_w, self.local_h),
                                              (self.full_w, self.full_h))

            self.pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                          self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.lmb[e, 0]:self.lmb[e, 1],
                                    self.lmb[e, 2]:self.lmb[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()
    
    def _init_map_and_pose_for_env(self, e):
        self.full_map[e].fill_(0.)
        self.full_pose[e].fill_(0.)
        self.full_pose[e, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[e].cpu().numpy()
        self.pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                          (self.local_w, self.local_h),
                                          (self.full_w, self.full_h))
        self.pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                      self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
        self.local_pose[e] = self.full_pose[e] - \
            torch.from_numpy(self.origins[e]).to(self.device).float()
        
    def reset_map_and_pose(self):
        for e in range(self.num_scenes):
            self._init_map_and_pose_for_env(e)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        '''
        以agent为中心, local map不超出full map的范围
        '''
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        self.local_w = local_w
        self.local_h = local_h
        self.full_w = full_w
        self.full_h = full_h

        return [gx1, gx2, gy1, gy2]


    def _update_next_view_local(self, local_map, local_pose):
        '''
        '''
        # update the full and local maps; 
        for e in range(self.num_scenes):

            # 用更新后的local map更新full map
            self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                local_map[e]
            self.full_pose[e] = local_pose[e] + \
                torch.from_numpy(self.origins[e]).to(self.device).float()

            locs = self.full_pose[e].cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            # 在full map上重新选agent为中心的local map
            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),       # 新的full pose下的local map的边界
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))
            self.pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                            self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

            local_map[e] = self.full_map[e, :,
                                    self.lmb[e, 0]:self.lmb[e, 1],
                                    self.lmb[e, 2]:self.lmb[e, 3]]
            local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()
        
        return local_map, local_pose

    def update_local_map(self, local_map):
        self.local_map = local_map

    def update_semantic_map(self, obs, infos):

        poses = torch.from_numpy(np.asarray(
                [infos[env_idx]['sensor_pose'] for env_idx
                in range(self.num_scenes)])
            ).float().to(self.device)
        
        # update 0: obstacle 1: explored 4...: semantic
        # agent当前观察到的自我中心的map
        _, local_map, _, local_pose = \
            self.semantic_map(obs, poses, self.local_map, self.local_pose)
        
        # update 2-3: curr and past maps
        locs = local_pose.cpu().numpy()
        self.pose_inputs[:, :3] = locs + self.origins
        local_map[:, 2, :, :].fill_(0.)
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]
            local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        
        local_map, local_pose = self._update_next_view_local(local_map, local_pose)

        # update 
        self.local_map = local_map
        self.local_pose = local_pose

        return local_map, local_pose
    
    def sum_of_semantic_map(self):
        # get semantic channels: 4:
        semantic_maps = self.full_map[:, 4:10, ...]   # num_scenes, num_semantic, size_w, size_h

        # sum of semantic objects
        sum_res = semantic_maps.sum(-1).sum(-1).sum(-1)  # num_scenes

        return sum_res
    
    def get_explore_area(self, explo_area):
        '''
        explo_area: num_scenes, 
        '''
        for e in range(self.num_scenes):
            explored_area = self.full_map[e, 1].sum(1).sum(0)
            explo_area[e] = explored_area
            explo_area[e] *= (self.args.map_resolution / 100.)**2  # to m^2
        return explo_area
    
if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:1"
    maps = Maps_Env(args)

    envs = make_vec_envs(args)      
    obs, infos = envs.reset()

    print(f"obs shape: {obs.shape}")
    local_map, local_pose = maps.update_semantic_map(obs, infos)
    print(f"local map shape: {local_map.shape}")
    print(f"local pose: {local_pose}")

    sum_rew  = maps.sum_of_semantic_map()
    print(f"all reward: {sum_rew}")

    expl_area = torch.zeros(2)
    expl_area = maps.get_explore_area(expl_area)
    print(f" explore area: {expl_area}")
