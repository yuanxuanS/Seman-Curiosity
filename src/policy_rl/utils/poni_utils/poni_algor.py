from .model_pf import RL_Policy
from src.policy_rl.utils.storage import GlobalRolloutStorage
import gym
import torch
import torch.nn as nn
import numpy as np
import src.policy_rl.envs.utils.pose as pu

class PONI:
    def __init__(self, args, num_scenes, device):
        self.num_scenes = num_scenes
        self.args = args
        
        self.g_policy = RL_Policy(args, args.pf_model_path).to(device)
        self.g_policy.eval()
        
        # Calculating full and local map sizes
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        self.local_w = local_w = int(full_w / args.global_downscaling)
        self.local_h = local_h = int(full_h / args.global_downscaling)


        # Global policy observation space
        self.ngc = ngc = 8 + args.poni_num_sem_categories
        self.es = es = 2
        g_observation_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")
        # Global policy action space
        g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

        self.g_rollouts = GlobalRolloutStorage(
            args.poni_num_global_steps,
            num_scenes,
            g_observation_space.shape,
            g_action_space,
            self.g_policy.rec_state_size,
            es,
        ).to(device)
        
        self.g_masks = torch.ones(num_scenes).float().to(device)
        # 规划周期
        self.replan_step = 50
        self.replan = [True]*self.num_scenes
        self.goals = [None]*self.num_scenes
        self.counts = [0]*self.num_scenes
        
        from torchvision import transforms
        from PIL import Image
        self.resize = transforms.Compose([transforms.Resize((self.local_h*self.args.global_downscaling,
                                                        self.local_w*self.args.global_downscaling),
                               interpolation=Image.NEAREST)])
    
    def reset(self):
        pass
        
    def get_global_goals(self, 
                   local_map, 
                   full_map, 
                   local_pose, 
                   vis_inputs, 
                   infos, 
                   ):
        
        # update replan
        for e in range(self.num_scenes):
            if self.goals[e] is None:
                self.replan[e] = True
            else:
                ## replan if get close to goal
                
                # full goal
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 =vis_inputs[e]['pose_pred']
                # gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                r, c = start_y, start_x
                start = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]
                start = pu.threshold_poses(start, local_map[e].shape)
                
                goal_r, goal_c = self.goals[e][0], self.goals[e][1]
                if abs(goal_c - start[1]) < 10 and abs(goal_r - start[0]) < 10:     # for grid distance
                    self.replan[e] = True
                    print(f"get in long term goal {self.replan[e]}")
                else:
                    self.replan[e] = self.counts[e] >= self.replan_step       # if long time
                    
                if self.replan[e]:
                    self.counts[e] = 0 
                else:
                    self.counts[e] += 1  
        
        # inference goals
        global_input = torch.zeros(self.num_scenes, self.ngc, 
                                   self.local_w*self.args.global_downscaling,
                                   self.local_h*self.args.global_downscaling)
        global_orientation = torch.zeros(self.num_scenes, 1).long()
        extras = torch.zeros(self.num_scenes, self.es)
        
        locs = local_pose.cpu().numpy()
        for e in range(self.num_scenes):
            global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
        
        # poni中full -> local的downscaling=1, local=480*480
        
        global_input[:, 0:4, :, :] = self.resize(local_map[:, 0:4, :, :].detach())
        global_input[:, 4:8, :, :] = nn.MaxPool2d(1)(
            full_map[:, 0:4, :, :]
        )
        # 分配到poni定义的各个类别： 
        orig_sem_map = self.resize(local_map[:, 4:, :, :].detach())      # 最后通道为辅助，非语义
        global_input[:, 8:10, :, :] = orig_sem_map[:, :2, :, :]     # chair, couch
        global_input[:, 11:13, :, :] = orig_sem_map[:, 3:5, :, :]   # bed, toilet
        global_input[:, 17, :, :] = orig_sem_map[:, 3, :, :]    # refrigerator
        global_input[:, -1, :, :] = orig_sem_map[:, -1, :, :]
        # global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
        goal_cat_id = torch.from_numpy(
            np.asarray([infos[env_idx]["goal_cat_id"] for env_idx in range(self.num_scenes)])
        )

        extras = torch.zeros(self.num_scenes, self.es)
        extras[:, 0] = global_orientation[:, 0]
        extras[:, 1] = goal_cat_id
    
        # if start:
        #     self.g_rollouts.obs[0].copy_(global_input)
        #     self.g_rollouts.extras[0].copy_(extras)
        # else:
        
        agent_locations = []
        for e in range(self.num_scenes):
            pose_pred = vis_inputs[e]['pose_pred']
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            map_r, map_c = start_y, start_x
            map_loc = [
                int(map_r * 100.0 / self.args.map_resolution - gx1),
                int(map_c * 100.0 / self.args.map_resolution - gy1),
            ]
            map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
            agent_locations.append(map_loc)
        
            
        g_obs = global_input.to(local_map.device)  # g_rollouts.obs[g_step]
        fmm_dists = None
        ego_agent_poses = None
        unk_map = 1.0 - local_map[:, 1, :, :]
        # Sample long-term goal from global policy
        g_value, g_action, g_action_log_prob, g_rec_states, prev_pfs = self.g_policy.act(
            g_obs,
            None,  # g_rollouts.rec_states[g_step],
            self.g_masks.to(g_obs.device),  # g_rollouts.masks[g_step],
            extras=extras.to(g_obs.device),  # g_rollouts.extras[g_step],
            extra_maps={
                "dmap": fmm_dists,
                "umap": self.resize(unk_map),
                # "pfs": prev_pfs,
                "agent_locations": agent_locations,
                "ego_agent_poses": ego_agent_poses,
            },
            deterministic=False,
        )
        
        if not self.g_policy.has_action_output:
            cpu_actions = g_action.cpu().numpy()
            if len(cpu_actions.shape) == 2:  # (B, 2) XY locations
                global_goals = [
                    [int(action[0] * self.local_w), int(action[1] * self.local_h)]
                    for action in cpu_actions
                ]
                global_goals = [
                    [min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                    for x, y in global_goals
                ]
            else:
                assert len(cpu_actions.shape) == 3  # (B, H, W) action maps
                global_goals = None
        
        # convert to full goal
        for e in range(self.num_scenes):
            pose_pred = vis_inputs[e]['pose_pred']
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            goal_c, goal_r = global_goals[e][0], global_goals[e][1]
            goal_c += gy1
            goal_r += gx1
            global_goals[e] = pu.threshold_poses([goal_r, goal_c], full_map[e].shape)
        
        # Update long-term goal if target object is found
        # found_goal = [0 for _ in range(self.num_scenes)]
        # goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]
        # if not self.g_policy.has_action_output:
        #     # Ignore goal and use nearest frontier baseline if requested
        #     if not self.args.use_nearest_frontier:
        #         for e in range(self.num_scenes):
        #             if global_goals is not None:
        #                 goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
        #             else:
        #                 goal_maps[e][:, :] = cpu_actions[e]
        #     else:
        #         # for e in range(num_scenes):
        #         #     fmap = frontier_maps[e].cpu().numpy()
        #         #     goal_maps[e][fmap] = 1
        #         pass        # 用不到
        
        # 如果找到目标物体则作为goal
        # for e in range(self.num_scenes):
        #     cn = infos[e]["goal_cat_id"] + 4
        #     cat_semantic_map = local_map[e, cn, :, :]

        #     if cat_semantic_map.sum() != 0.0:
        #         cat_semantic_map = cat_semantic_map.cpu().numpy()
        #         cat_semantic_scores = cat_semantic_map
        #         cat_semantic_scores[cat_semantic_scores > 0] = 1.0
        #         goal_maps[e] = cat_semantic_scores
        #         found_goal[e] = 1
        
        self.goals = [g_goal if self.replan[e] else self.goals[e] for g_goal in global_goals] 
        return self.goals
             
    def update_vis(self, 
                   planner_inputs, 
                   goal_maps, 
                   l_step):     # no call
        pf_visualizations = None
        if self.args.visualize or self.args.print_images:
            pf_visualizations = self.g_policy.visualizations
        for e, p_input in enumerate(planner_inputs):
            p_input["goal"] = goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = l_step == self.args.num_local_steps - 1
            
            if self.args.visualize or self.args.print_images:
                # local_map[e, -1, :, :] = 1e-5
                # p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                p_input["pf_pred"] = pf_visualizations[e]
                # obs[e, -1, :, :] = 1e-5
                # p_input["sem_seg"] = obs[e, 4:].argmax(0).cpu().numpy()
        return planner_inputs