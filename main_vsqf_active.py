
from src.policy_rl.arguments import get_args
import torch
from torchvision import transforms
import numpy as np
import os
import logging
from collections import deque, defaultdict
import gym
import time
from datetime import datetime
from src.policy_rl.envs import make_vec_envs
from src.policy_rl.maps import Maps_Env
from src.policy_rl.vsqf_maps import Vsqf_Maps_Env
from src.policy_rl.utils.storage import GlobalRolloutStorage
from src.policy_rl.model import RL_Policy
from  src.policy_rl import algo 
from src.policy_rl.baseline_frontier import Frontier
import cv2
import json
from src.policy_rl.agents.utils.vsqf_prediction import Vsqf_pred
from src.policy_rl.agents.utils.orient_prediction import Orient_pred
from PIL import Image
from vqf_train import visualize
from vsqf_heuristic import vsqf_heuristic
from src.vqf_constants import target_coco_categories
def main():
    args = get_args()
    
    # seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    log_name = 'eval.log' if args.eval else 'train.log' 
    logging.basicConfig(
        filename=log_dir + log_name,
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")   # 训练的gpu

    

    best_l_reward = -np.inf

    if args.eval:
        # TODO: 增加一些online指标
        # episode_done = []
        # for _ in range(args.num_processes):
        #     episode_done.append(deque(maxlen=num_episodes))
        pass
    else:
        pass # TODO: 
    
    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    l_episode_rewards = []
    per_step_l_rewards = deque(maxlen=1000)
    per_step_rewards = deque(maxlen=1000)
    

    # Init environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      

    torch.set_grad_enabled(False)

    ## Initializing Maps
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    maps = Maps_Env(args)
    
    ## Initializing VSQF Maps
    vsqf_maps = Vsqf_Maps_Env(args)
    
    ## initialize vsqf, orient predictor
    magnify, magnify_num = args.magnify, args.magnify_num
    vsqf_pred = Vsqf_pred(device, magnify, magnify_num)
    orient_pred = Orient_pred(device)
    ## init vsqf algorithm
    vsqf_heu = vsqf_heuristic(args, num_scenes, device)
    vsqf_heu.reset()
    
    
    # init current obj map for goal computation
    curr_object_maps = [np.ones((maps.full_w, maps.full_h))] * num_scenes
    
    
    ## active cam policy
    l_observation_space = envs.get_obs_space()[0]  # TODO: VectorEnv's func
    l_action_space = envs.get_action_space()[0]
    l_hidden_size = args.local_hidden_size
    camera_policy = RL_Policy(l_observation_space.shape, l_action_space,
                        model_type=1,
                        base_kwargs={'recurrent': args.use_recurrent_local,
                                    'hidden_size': l_hidden_size,
                                    'num_sem_categories': args.num_sem_categories
                                    }).to(device)
    # Storage:
    l_rollouts = GlobalRolloutStorage(args.num_local_steps,
                                    num_scenes, l_observation_space.shape,
                                    l_action_space, camera_policy.rec_state_size,
                                    0).to(device)
    extras = torch.zeros(num_scenes, 0)
    res = transforms.Compose(
            [
            # transforms.ToPILImage(),
             transforms.Resize((args.camera_frame_height, args.camera_frame_width),
                               interpolation=Image.NEAREST)])
    #  l_masks, not used. episode length不同时使用
    l_masks = torch.ones(num_scenes).float().to(device)
    
    if args.load != "0":
        print("Loading model {}".format(args.load))
        logging.info("Loading model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        camera_policy.load_state_dict(state_dict)
    if args.eval:
        camera_policy.eval()
    ## ------------------start------------------
    obs, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?
    
    if args.explore_algor == "gt":
        target_locs = envs.get_target_rel_loc()     # 在初始agent坐标系中的dx dy
        vsqf_heu.explore_policy.set_goals(target_locs)
        
        rest_goal = [info['rest_goal'] for info in infos] 
        vsqf_heu.explore_policy.update_goal_deque(rest_goal)
    # update map
    local_map, local_pose = maps.update_semantic_map(obs, infos)
    full_pose = maps.full_pose
    
    # inference vsqf and azimuth
    rgb_objs = [infos[env_idx]['rgb_obj'] for env_idx in range(num_scenes)]
    rgb_objs = [Image.fromarray(rgb_obj.astype(np.uint8)) for rgb_obj in rgb_objs]
    rgb_objs_ = vsqf_pred.preprocess(rgb_objs)
    vsqf = vsqf_pred.inference(rgb_objs_)
    orient_data = orient_pred.pred_orient_multi(rgb_objs)
    azimuth, confidence = orient_data
    
    # check if find goal
    find_goal = torch.tensor([infos[env_idx]['find_goal'] for env_idx in range(num_scenes)])
    find_goal = find_goal.to(vsqf.device)
    vsqf = find_goal[:, None, None] * vsqf
    azimuth = find_goal * azimuth
    
    # update vsqf maps
    local_vsqf_map, _ = vsqf_maps.update_vsqf_map(infos, vsqf, azimuth)
    full_vsqf_map = vsqf_maps.full_map    

    # update current obj map
    full_map = maps.full_map
    curr_full_map = maps.curr_full_map
    for e in range(num_scenes):
        if find_goal[e]:
            curr_full_map[e, -1, :, :] = 1e-5
            curr_object_map = curr_full_map[e, 4:, :, :].argmax(0).cpu().numpy()
        else:
            if not infos[e]['sample_stage']:
                curr_object_map = np.ones_like(curr_object_maps[0])
            else:
                curr_object_map = curr_object_maps[e]
        curr_object_maps[e] = curr_object_map
    
    finish_sample = [infos[env_idx]['finished'] for env_idx in range(num_scenes)]
    timestep = [infos[env_idx]['time'] for env_idx in range(num_scenes)]
    patch = [True if timestep[e] == 0 else False for e in range(num_scenes)]
    maps.patch_agent_region(patch)
    
    # for visualize
    vis_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(vis_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = maps.get_all_pose()[e]
        
        p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = maps.get_all_pose()[e]

        local_map[e, -1, :, :] = 1e-5       # 有物体时，为了argmax时不选最后通道
        p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                            ].argmax(0).cpu().numpy()   # 如果无object，选最后一个通道
        full_map[e, -1, :, :] = 1e-5
        p_input['sem_map_pred_full'] = full_map[e, 4:, :, :].argmax(0).cpu().numpy()
        
        p_input['vsqf_map'] = local_vsqf_map[e, :, :, :].cpu().numpy()
        p_input['vsqf_map_full'] = full_vsqf_map[e, :, :, :].cpu().numpy()
        p_input['object_map_full'] = curr_object_maps[e]
    
    
    
    if args.agent == "random":
        camera_action = np.random.randint(0, 5, num_scenes)
                
        # select goal according to vsqf map
        update_vis = [info['sample_stage']*(info['sample_step'] % 5 == 1) for info in infos]
        for e, p_input in enumerate(vis_inputs):
            p_input['sample_step'] = infos[e]['sample_step']
            p_input['sample_stage'] = infos[e]['sample_stage']
            p_input['depth'] = infos[e]['depth']
            p_input['time'] = infos[e]['time']
            p_input['camera_stage'] = infos[e]['camera_stage']
            p_input['camera_step'] = infos[e]['camera_step']
            p_input['invalid_goal'] = infos[e]['invalid_goal']
        goals = vsqf_heu.get_random_region(vsqf_maps.full_map, vis_inputs, update_vis_map=update_vis)
        for e, p_input in enumerate(vis_inputs):
            if infos[e]['sample_stage']:
                p_input["frontier_goal"] = goals[e]
            
        # return action with planner
        if args.explore_algor == "frontier" or args.explore_algor == "gt":
            l_action = vsqf_heu.get_actions(vis_inputs, camera_action)
        elif args.explore_algor == "poni":
            l_action = vsqf_heu.get_actions(vis_inputs, camera_action,
                                            {'local_map':local_map, 
                                            'full_map':full_map,
                                            'local_pose':local_pose, 
                                            'infos':infos})
    elif args.agent == "frontier":
        nav_policy = Frontier(args)
        nav_policy.reset(num_scenes)
        for e, p_input in enumerate(vis_inputs):
            p_input['depth'] = infos[e]['depth']
            p_input['time'] = infos[e]['time']
            p_input['sample_stage'] = False
        l_action, goals, short_time_goals = nav_policy.get_actions(vis_inputs)        
        for e, p_input in enumerate(vis_inputs):
            # if args.visualize or args.print_images:
            p_input["frontier_goal"] = goals[e]
            # p_input["short_time_goal"] = short_time_goals[e]
            
    elif args.agent == "vsqf_heuristic":
        # run camera policy
        camera_action = np.zeros(num_scenes)
        for e in range(num_scenes):
            if infos[e]['camera_stage']:
                if infos[e]['camera_step'] == 0:
                    # camera policy
                    l_input_env = infos[e]['cam_obs'][e, :3, ...]
                    l_input_env = res(l_input_env)
                    l_rollouts.obs[0][e].copy_(l_input_env)   #
                    ind = 0
                else:
                    ind = infos[e]['camera_step']
                # Run camera policy
                l_value, camera_action_, l_action_log_prob, l_rec_states = \
                    camera_policy.act(
                        l_rollouts.obs[ind],
                        l_rollouts.rec_states[ind],
                        l_rollouts.masks[ind],
                        extras=l_rollouts.extras[ind],
                        deterministic=False
                    )
                camera_action[e] = camera_action_.cpu().numpy()
            else:
                l_rollouts.reset()
            
        # select goal according to vsqf map
        update_vis = [info['sample_stage']*(info['sample_step'] % 5 == 1) for info in infos]
        for e, p_input in enumerate(vis_inputs):
            p_input['sample_step'] = infos[e]['sample_step']
            p_input['sample_stage'] = infos[e]['sample_stage']
            p_input['depth'] = infos[e]['depth']
            p_input['time'] = infos[e]['time']
            p_input['camera_stage'] = infos[e]['camera_stage']
            p_input['camera_step'] = infos[e]['camera_step']
            p_input['invalid_goal'] = infos[e]['invalid_goal']
            
        goals = vsqf_heu.get_best_region(vsqf_maps.full_map, vis_inputs, update_vis_map=update_vis)
        for e, p_input in enumerate(vis_inputs):
            if infos[e]['sample_stage']:
                p_input["frontier_goal"] = goals[e]
        
        # return action with planner
        if args.explore_algor == "frontier" or args.explore_algor == "gt":
            l_action = vsqf_heu.get_actions(vis_inputs, camera_action)
        elif args.explore_algor == "poni":
            l_action = vsqf_heu.get_actions(vis_inputs, camera_action, 
                                            {'local_map':local_map, 
                                            'full_map':full_map,
                                            'local_pose':local_pose, 
                                            'infos':infos}
                                            )
    # transition:
    # pred instance, get semantic masks and step env: 
    obs, _, done, infos = envs.step_and_pre(l_action, vis_inputs, wait_env)
    l_action = torch.tensor(l_action)
    # update map
    local_map, local_pose = maps.update_semantic_map(obs, infos)
    full_pose = maps.full_pose
    
    # inference vsqf and azimuth
    rgb_objs = [infos[env_idx]['rgb_obj'] for env_idx in range(num_scenes)]
    rgb_objs = [Image.fromarray(rgb_obj.astype(np.uint8)) for rgb_obj in rgb_objs]
    rgb_objs_ = vsqf_pred.preprocess(rgb_objs)
    vsqf = vsqf_pred.inference(rgb_objs_)
    orient_data = orient_pred.pred_orient_multi(rgb_objs)
    azimuth, confidence = orient_data
    
    # check if find goal
    # find_goal = torch.tensor([False for _ in range(num_scenes)])
    find_goal = torch.tensor([infos[env_idx]['find_goal'] for env_idx in range(num_scenes)])
    find_goal = find_goal.to(vsqf.device)
    vsqf = find_goal[:, None, None] * vsqf
    azimuth = find_goal * azimuth
    
    local_vsqf_map, _ = vsqf_maps.update_vsqf_map(infos, vsqf, azimuth)
    full_vsqf_map = vsqf_maps.full_map
    

    
    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    logging.info("Start date and time: %s", start_datetime)
    
    l_reward = torch.zeros(num_scenes).to(device)
    last_scores = torch.zeros(num_scenes).to(device)

    
    torch.set_grad_enabled(False)

    print("Starting running")
    logging.info("Starting running")
    if not args.eval:
        print(f"training frames is {args.num_training_frames}")
        logging.info(f"training frames is {args.num_training_frames}")
    for step in range(args.num_training_frames // args.num_processes + 1):
        l_step = step % args.num_local_steps
        
        if finished.sum() == args.num_processes:    # eval over
            break
        
        # get reward: map change after state transition
        if args.agent == "vsqf_heuristic":
            for e in range(num_scenes):
                if infos[e]['camera_stage']:
                    
                    if infos[e]['camera_step'] == 0:
                        # camera policy
                        l_input_env = infos[e]['cam_obs'][:3, ...]
                        l_input_env = res(torch.from_numpy(l_input_env))
                        l_rollouts.obs[0][e].copy_(l_input_env)   #
                    else:
                        # update policy input with next state
                        l_input = infos[e]['cam_obs'][:3, ...]       # rgb
                        l_input = res(torch.from_numpy(l_input))
                        # Add samples to local policy storage
                        l_rollouts.insert(
                                    l_input, l_rec_states,      # state_t+1
                                    l_action, l_action_log_prob, l_value,   # action, reward_t
                                    l_reward, l_masks, extras
                                )
        # ------------------------------------------------------------------ 
        # update local input, next state
        # locs = full_pose.cpu().numpy()

        for e, x in enumerate(done):
            wait_env[e] = 1 if x else wait_env[e]
            
        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        if wait_env.sum() == num_scenes:
            r_ = np.mean(l_reward.cpu().numpy())
            print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
            logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
            l_episode_rewards.append(r_)

            l_reward = torch.zeros(num_scenes).to(device)
            last_scores = l_reward
            
            if args.eval:
                # for e, x in enumerate(done):    # if done, maps from new obs
                #     episode_done[e].append(True)
                # if len(episode_done[e]) == num_episodes:
                for e in range(num_scenes):
                    if finish_sample[e]:
                        finished[e] = 1
                        pass

            if args.agent == "frontier":
                nav_policy.reset(num_scenes)


        if step%50 == 0:
            maps.filter_obstacle_map()
            for e, p_input in enumerate(vis_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()                
                p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
            
        
        # update current obj map
        full_map = maps.full_map
        curr_full_map = maps.curr_full_map
        for e in range(num_scenes):
            if find_goal[e]:
                curr_full_map[e, -1, :, :] = 1e-5
                curr_object_map = curr_full_map[e, 4:, :, :].argmax(0).cpu().numpy()
            else:
                if not infos[e]['sample_stage']:
                    curr_object_map = np.ones_like(curr_object_maps[0])
                else:
                    curr_object_map = curr_object_maps[e]
            curr_object_maps[e] = curr_object_map  
        
        timestep = [infos[env_idx]['time'] for env_idx in range(num_scenes)]
        finish_sample = [infos[env_idx]['finished'] for env_idx in range(num_scenes)]
        patch = [True if timestep[e] == 0 else False for e in range(num_scenes)]
        maps.patch_agent_region(patch)
        
        
        
        vis_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(vis_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]

            p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]

            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()
            full_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
                                                    ].argmax(0).cpu().numpy()  
            p_input['vsqf_map'] = local_vsqf_map[e, :, :, :].cpu().numpy()
            p_input['vsqf_map_full'] = full_vsqf_map[e, :, :, :].cpu().numpy()                  
            p_input['object_map_full'] = curr_object_maps[e]
        
        # Sample next action
        if args.agent == "random":
            camera_action = np.random.randint(0, 5, num_scenes)
            
            # select goal according to vsqf map
            update_vis = [info['sample_stage']*(info['sample_step'] % 5 == 1) for info in infos]
            for e, p_input in enumerate(vis_inputs):
                p_input['sample_step'] = infos[e]['sample_step']
                p_input['sample_stage'] = infos[e]['sample_stage']
                p_input['depth'] = infos[e]['depth']
                p_input['time'] = infos[e]['time']
                p_input['camera_stage'] = infos[e]['camera_stage']
                p_input['camera_step'] = infos[e]['camera_step']
                p_input['invalid_goal'] = infos[e]['invalid_goal']
            goals = vsqf_heu.get_random_region(vsqf_maps.full_map, vis_inputs, update_vis_map=update_vis)
            for e, p_input in enumerate(vis_inputs):
                if infos[e]['sample_stage']:
                    p_input["frontier_goal"] = goals[e]
                
            # return action with planner
            if args.explore_algor == "frontier" or args.explore_algor == "gt":
                l_action = vsqf_heu.get_actions(vis_inputs, camera_action)
            elif args.explore_algor == "poni":
                l_action = vsqf_heu.get_actions(vis_inputs, camera_action, 
                                                {'local_map':local_map, 
                                            'full_map':full_map,
                                            'local_pose':local_pose, 
                                            'infos':infos})
        if args.agent == "frontier":  # must be after updating vis_inputs
            for e, p_input in enumerate(vis_inputs):
                p_input['depth'] = infos[e]['depth']
                p_input['time'] = infos[e]['time']
                p_input['sample_stage'] = False
            l_action, goals, short_time_goals = nav_policy.get_actions(vis_inputs)        
            for e, p_input in enumerate(vis_inputs):
                p_input["frontier_goal"] = goals[e]
                # p_input["short_time_goal"] = short_time_goals[e]
                
        if args.agent == "vsqf_heuristic":
            # run camera policy
            camera_action = np.zeros(num_scenes)
            for e in range(num_scenes):
                if infos[e]['camera_stage']:
                    if infos[e]['camera_step'] == 0:
                        idx = 0
                        
                    else:
                        idx = infos[e]['camera_step']
                    # Run camera policy
                    l_value, camera_action_, l_action_log_prob, l_rec_states = \
                        camera_policy.act(
                            l_rollouts.obs[idx],
                            l_rollouts.rec_states[idx],
                            l_rollouts.masks[idx],
                            extras=l_rollouts.extras[idx],
                            deterministic=False
                        )
                    camera_action[e] = camera_action_
                else:
                    l_rollouts.reset()

            
            
            # select goal according to vsqf map
            update_vis = [info['sample_stage']*(info['sample_step'] % 5 == 1) for info in infos]
            for e, p_input in enumerate(vis_inputs):
                p_input['sample_step'] = infos[e]['sample_step']
                p_input['sample_stage'] = infos[e]['sample_stage']
                p_input['depth'] = infos[e]['depth']
                p_input['time'] = infos[e]['time']
                p_input['camera_stage'] = infos[e]['camera_stage']
                p_input['camera_step'] = infos[e]['camera_step']
                p_input['invalid_goal'] = infos[e]['invalid_goal']
            goals = vsqf_heu.get_best_region(vsqf_maps.full_map, vis_inputs, update_vis_map=update_vis)
            for e, p_input in enumerate(vis_inputs):
                if infos[e]['sample_stage']:
                    p_input["frontier_goal"] = goals[e]
            
            if args.explore_algor == "gt":
                rest_goal = [info['rest_goal'] for info in infos] 
                vsqf_heu.explore_policy.update_goal_deque(rest_goal)
            # return action with planner
            if args.explore_algor == "frontier" or args.explore_algor == "gt":
                l_action = vsqf_heu.get_actions(vis_inputs, camera_action)
            elif args.explore_algor == "poni":
                l_action = vsqf_heu.get_actions(vis_inputs, camera_action, 
                                                {'local_map':local_map, 
                                            'full_map':full_map,
                                            'local_pose':local_pose, 
                                            'infos':infos})
        
        # transition: next state
        if wait_env.sum() == num_scenes:
            # if episode over, reset maps
            for e in range(num_scenes):
                maps._init_map_and_pose_for_env(e)
                vsqf_maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
            vsqf_heu.reset()
                
            obs, infos = envs.reset()
            if args.explore_algor == "gt":
                target_locs = envs.get_target_rel_loc()     # 在初始agent坐标系中的dx dy
                vsqf_heu.explore_policy.set_goals(target_locs)

                rest_goal = [info['rest_goal'] for info in infos] 
                vsqf_heu.explore_policy.update_goal_deque(rest_goal)
        
            done  = np.array([False]*num_scenes)
            wait_env = np.zeros((args.num_processes))
            curr_object_maps = [np.ones_like(full_vsqf_map[0].cpu().numpy())] * num_scenes
            
        else:
            # pred instance, get semantic masks and step env
            obs, _, done, infos = envs.step_and_pre(l_action, vis_inputs, wait_env)    # if done ,envs.reset, obs are ones after reset
        l_action = torch.tensor(l_action)
        
        # update map
        local_map, local_pose = maps.update_semantic_map(obs, infos)
        full_pose = maps.full_pose
        
        # inference vsqf and azimuth
        rgb_objs = [infos[env_idx]['rgb_obj'] for env_idx in range(num_scenes)]
        rgb_objs = [Image.fromarray(rgb_obj.astype(np.uint8)) for rgb_obj in rgb_objs]
        rgb_objs_ = vsqf_pred.preprocess(rgb_objs)
        vsqf = vsqf_pred.inference(rgb_objs_)
        orient_data = orient_pred.pred_orient_multi(rgb_objs)
        azimuth, confidence = orient_data
        
        # check if find goal
        # find_goal = torch.tensor([False for _ in range(num_scenes)])
        find_goal = torch.tensor([infos[env_idx]['find_goal'] for env_idx in range(num_scenes)])
        find_goal = find_goal.to(vsqf.device)
        vsqf = find_goal[:, None, None] * vsqf
        azimuth = find_goal * azimuth
        
        # update vsqf maps
        local_vsqf_map, _ = vsqf_maps.update_vsqf_map(infos, vsqf, azimuth)
        full_vsqf_map = vsqf_maps.full_map
                
        
        
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging: TODO
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            # log += "\n\tRewards:"

            # if len(per_step_rewards) > 0:
            #     log += " ".join([
            #         " per step mean/med/min/max, rew:",
            #         "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
            #             np.mean(per_step_rewards),
            #             np.median(per_step_rewards),
            #             np.min(per_step_rewards),
            #             np.max(per_step_rewards))
            #     ])

            # log += "\n\tLosses:"
            # if len(l_value_losses) > 0 and not args.eval:
            #     log += " ".join([
            #         " Policy Loss value/action/dist:",
            #         "{:.3f}/{:.3f}/{:.3f},".format(
            #             np.mean(l_value_losses),
            #             np.mean(l_action_losses),
            #             np.mean(l_dist_entropies))
            #     ])
                
            # if done[0]:
            #     if len(l_episode_rewards) > 0:
            #         log += " ".join([
            #         " episode mean/med/min/max, rew:",
            #         "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
            #             np.mean(l_episode_rewards),
            #             np.median(l_episode_rewards),
            #             np.min(l_episode_rewards),
            #             np.max(l_episode_rewards))
            #         ])
                
            print(log)
            logging.info(log)

        # ------------------------------------------------------------------
    # Print and save model performance numbers during evaluation: TODO
    # with open('{}/{}_episode_rewards.json'.format(
    #         dump_dir, args.split), 'w') as f:
    #     json.dump(l_episode_rewards, f)
    m = np.array(l_episode_rewards).mean()
    print(f"all episode rewards: {l_episode_rewards}, mean is {m}")
    logging.info(f"all episode rewards: {l_episode_rewards}, mean is {m}")
    np.savez('{}/{}_episode_rewards.npz'.format(
            dump_dir, args.split), episode_reward=l_episode_rewards)
    
    if args.eval:
        print("Dumping eval details...")
        

        
if __name__ == "__main__":
    main()