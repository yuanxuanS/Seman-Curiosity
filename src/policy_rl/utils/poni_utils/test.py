from .model_pf import RL_Policy
from semexp.arguments import get_args
import torch
from src.policy_rl.utils.storage import GlobalRolloutStorage
import gym
import numpy as np
import src.policy_rl.envs.utils.pose as pu

if __name__ == "__main__":
    args = get_args()
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    num_scenes = args.num_processes
    
    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")
    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)


    # Global policy
    g_policy = RL_Policy(args, args.pf_model_path).to(device)
    
    if args.eval:
        g_policy.eval()
    
    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps,
        num_scenes,
        g_observation_space.shape,
        g_action_space,
        g_policy.rec_state_size,
        es,
    ).to(device)
    
    # TODO
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    extras = torch.zeros(num_scenes, es)
    
    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)
    
    fmm_dists = None
    ego_agent_poses = None
    
    
    agent_locations = []
    for e in range(num_scenes):
        pose_pred = planner_pose_inputs[e]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_r, map_c = start_y, start_x
        map_loc = [
            int(map_r * 100.0 / args.map_resolution - gx1),
            int(map_c * 100.0 / args.map_resolution - gy1),
        ]
        map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
        agent_locations.append(map_loc)
                
    # g_obs = g_rollouts.obs[0]
    g_obs = global_input.to(local_map.device)  # g_rollouts.obs[g_step]

    # Reinitialize variables when episode ends
    l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
    g_masks *= l_masks
        
    unk_map = 1.0 - local_map[:, 1, :, :]
    # Sample long-term goal from global policy
    g_value, g_action, g_action_log_prob, g_rec_states, prev_pfs = g_policy.act(
        g_obs,
        None,  # g_rollouts.rec_states[g_step],
        g_masks.to(g_obs.device),  # g_rollouts.masks[g_step],
        extras=extras.to(g_obs.device),  # g_rollouts.extras[g_step],
        extra_maps={
            "dmap": fmm_dists,
            "umap": unk_map,
            "pfs": prev_pfs,
            "agent_locations": agent_locations,
            "ego_agent_poses": ego_agent_poses,
        },
        deterministic=False,
    )
    
    # ------------------------------------------------------------------
    

    if not g_policy.has_action_output:
        cpu_actions = g_action.cpu().numpy()
        if len(cpu_actions.shape) == 2:  # (B, 2) XY locations
            global_goals = [
                [int(action[0] * local_w), int(action[1] * local_h)]
                for action in cpu_actions
            ]
            global_goals = [
                [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                for x, y in global_goals
            ]
        else:
            assert len(cpu_actions.shape) == 3  # (B, H, W) action maps
            global_goals = None
    
    # Update long-term goal if target object is found
    found_goal = [0 for _ in range(num_scenes)]
    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    
    if not g_policy.has_action_output:
        # Ignore goal and use nearest frontier baseline if requested
        if not args.use_nearest_frontier:
            for e in range(num_scenes):
                if global_goals is not None:
                    goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                else:
                    goal_maps[e][:, :] = cpu_actions[e]
        else:
            # for e in range(num_scenes):
            #     fmap = frontier_maps[e].cpu().numpy()
            #     goal_maps[e][fmap] = 1
            pass        # 用不到
                    
    for e in range(num_scenes):
        cn = infos[e]["goal_cat_id"] + 4
        cat_semantic_map = local_map[e, cn, :, :]

        if cat_semantic_map.sum() != 0.0:
            cat_semantic_map = cat_semantic_map.cpu().numpy()
            cat_semantic_scores = cat_semantic_map
            cat_semantic_scores[cat_semantic_scores > 0] = 1.0
            goal_maps[e] = cat_semantic_scores
            found_goal[e] = 1
    
    # Take action and get next observation
    planner_inputs = [{} for e in range(num_scenes)]
    pf_visualizations = None
    if args.visualize or args.print_images:
        pf_visualizations = g_policy.visualizations
    for e, p_input in enumerate(planner_inputs):
        p_input["goal"] = goal_maps[e]  # global_goals[e]
        p_input["new_goal"] = l_step == args.num_local_steps - 1
        
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
            p_input["pf_pred"] = pf_visualizations[e]
            obs[e, -1, :, :] = 1e-5
            p_input["sem_seg"] = obs[e, 4:].argmax(0).cpu().numpy()
            
    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)