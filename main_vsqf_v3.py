
from src.policy_rl.arguments import get_args
import torch
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
import torch.nn as nn

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

    #  l_masks, not used. episode length不同时使用
    l_masks = torch.ones(num_scenes).float().to(device)

    best_l_reward = -np.inf

    if args.eval:
        # TODO: 增加一些online指标
        episode_done = []
        for _ in range(args.num_processes):
            episode_done.append(deque(maxlen=num_episodes))
    else:
        pass # TODO: 
    
    finished = np.zeros((args.num_processes))

    l_episode_rewards = []
    l_episode_dis_r = []
    l_episode_vsqf_r = []
    per_step_l_rewards = deque(maxlen=1000)
    per_step_rewards = deque(maxlen=1000)
    per_step_dis_rewards = deque(maxlen=1000)
    per_step_vsqf_rewards = deque(maxlen=1000)
    
    l_value_losses = deque(maxlen=1000)
    l_action_losses = deque(maxlen=1000)
    l_dist_entropies = deque(maxlen=1000)

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      
    obs, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?

    torch.set_grad_enabled(False)

    # Initializing Maps
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    maps = Maps_Env(args)
    local_map, local_pose = maps.update_semantic_map(obs, infos)
    full_pose = maps.full_pose
    
    local_w, local_h = maps.local_w, maps.local_h
    
    # Initializing VSQF Maps
    vsqf_maps = Vsqf_Maps_Env(args)
    magnify, magnify_num = args.magnify, args.magnify_num
    vsqf_pred = Vsqf_pred(device, magnify, magnify_num)
    orient_pred = Orient_pred(device)
    
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
        # for pointnav
        p_input['time'] = infos[e]['time']
        p_input['depth'] = infos[e]['depth']
        
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5       # 有物体时，为了argmax时不选最后通道
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()   # 如果无object，选最后一个通道
            full_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred_full'] = full_map[e, 4:, :, :].argmax(0).cpu().numpy()
            
            p_input['vsqf_map'] = local_vsqf_map[e, :, :, :].cpu().numpy()
            p_input['vsqf_map_full'] = full_vsqf_map[e, :, :, :].cpu().numpy()

    if args.agent == "rl":
        # Local policy observation space
        es = 1      # extra size: orientation
        ngc = 4 + args.num_sem_categories + 1
        
        l_observation_space = gym.spaces.Box(0, 1,      
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')
        l_action_space = envs.get_action_space()[0]

        # local policy recurrent layer size
        l_hidden_size = args.local_hidden_size

        # Local policy: TODO
        l_policy = RL_Policy(l_observation_space.shape, l_action_space,
                            model_type=2,
                            base_kwargs={'recurrent': args.use_recurrent_local,
                                        'hidden_size': l_hidden_size,
                                        'num_sem_categories': args.num_sem_categories,  # TODO
                                        }).to(device)
        
        l_agent = algo.PPO(l_policy, args.clip_param, args.ppo_epoch,
                        args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)

        

        # Storage: 
        l_rollouts = GlobalRolloutStorage(args.num_local_steps,
                                        num_scenes, l_observation_space.shape,
                                        l_action_space, l_policy.rec_state_size,
                                        es).to(device)
        
        # load weights
        if args.load != "0":
            print("Loading model {}".format(args.load))
            logging.info("Loading model {}".format(args.load))
            state_dict = torch.load(args.load,
                                    map_location=lambda storage, loc: storage)
            l_policy.load_state_dict(state_dict)

        if args.eval:
            l_policy.eval()
    
        # Get local policy input
        local_input = torch.zeros(num_scenes, ngc, local_w, local_h)
        local_orientation = torch.zeros(num_scenes, 1).long()

        # local_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()  # local_map的 obstacle, explored, curr loc, past loc
        # # TODO: add vsqf
        # local_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        #     full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
        # local_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()     # local semantic map
        local_input[:, 0:4, :, :] = nn.MaxPool2d(args.global_downscaling)(
            full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
        local_input[:, 4:4+args.num_sem_categories, :, :] = nn.MaxPool2d(args.global_downscaling)(
            full_map[:, 4:, :, :]).detach()     # full_map的 semantic map
        local_input[:, 4+args.num_sem_categories, :, :] = nn.MaxPool2d(args.global_downscaling)(
            full_vsqf_map[:, 0, :, :]).detach()
        
        locs = local_pose.cpu().numpy()
        # locs = full_pose.cpu().numpy()      # 使用全局pose
        for e in range(num_scenes):
            local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            
        extras = torch.zeros(num_scenes, es)
        extras[:, 0] = local_orientation[:, 0]

        l_rollouts.obs[0].copy_(local_input)   # 
        l_rollouts.extras[0].copy_(extras)

        # Run Local policy
        l_value, l_action, l_action_log_prob, l_rec_states = \
            l_policy.act(
                l_rollouts.obs[0],
                l_rollouts.rec_states[0],
                l_rollouts.masks[0],
                extras=l_rollouts.extras[0],
                deterministic=False
            )
        l_action = l_action.cpu().numpy()
    
    elif args.agent == "random":
        l_action = np.random.randint(0, 3, num_scenes)
    elif args.agent == "frontier":
        l_policy = Frontier(args)
        l_policy.reset(num_scenes)
        l_action, goals, short_time_goals = l_policy.get_actions(vis_inputs)        
        for e, p_input in enumerate(vis_inputs):
            if args.visualize or args.print_images:
                p_input["frontier_goal"] = goals[e]
                p_input["short_time_goal"] = short_time_goals[e]
    # transition:
    # pred instance, get semantic masks and step env: 
    obs, distance_rewards, done, infos = envs.step_and_preprocess(l_action, vis_inputs)
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
    l_cumu_r = torch.zeros(num_scenes).to(device)
    l_cumu_dis_r = torch.zeros(num_scenes).to(device)
    l_cumu_vsqf_r = torch.zeros(num_scenes).to(device)
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
        if done[0]:     # maps are new obs, sum of map will be small, and get negative reward
            l_reward = last_scores
            
        else:
            sample_stage = torch.from_numpy(np.asarray(
                [infos[env_idx]['sample_stage'] for env_idx
                in range(num_scenes)])
            ).float().to(device)
            
            l_scores = torch.tensor(vsqf_maps.get_vsqf_score()).to(device)
            sample_reward = l_scores - last_scores
            sample_reward = sample_reward * sample_stage * 10
            # distance reward
            distance_rewards = distance_rewards.to(device)
            distance_rewards = distance_rewards * (1 - sample_stage)
            
            l_reward = sample_reward + distance_rewards
            # log
            l_cumu_dis_r += distance_rewards
            l_cumu_vsqf_r += sample_reward
            

            # log
        l_cumu_r += l_reward
        # ------------------------------------------------------------------ 
        # update local input, next state
        # locs = full_pose.cpu().numpy()
        locs = local_pose.cpu().numpy()
        
        if args.agent == "rl":
            for e in range(num_scenes):
                local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)   # 
                
            # local_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()  # local_map的 obstacle, explored, curr loc, past loc
            # # TODO: add vsqf
            # local_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
            #     full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
            # local_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()     # local semantic map

            local_input[:, 0:4, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
            local_input[:, 4:4+args.num_sem_categories, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_map[:, 4:, :, :]).detach()     # full_map的 semantic map
            local_input[:, 4+args.num_sem_categories, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_vsqf_map[:, 0, :, :]).detach()
            
            extras[:, 0] = local_orientation[:, 0]
            # print(f"input sxtras: {extras}")
        # Add samples to local policy storage
        reward = l_reward
        
        if args.agent == "rl":
            l_rollouts.insert(
                    local_input, l_rec_states,      # state_t+1
                    l_action, l_action_log_prob, l_value,   # action, reward_t
                    reward, l_masks, extras
                )
        last_scores = l_scores
        # 
        reward_mean = np.mean(reward.cpu().numpy())
        l_reward_mean = np.mean(l_reward.cpu().numpy())
        per_step_rewards.append(reward_mean)
        per_step_l_rewards.append(l_reward_mean)         
        if int(sample_stage.sum().cpu().numpy()) != num_scenes:
            per_step_dis_rewards.append(torch.sum(distance_rewards).cpu().numpy() / (num_scenes - sample_stage.sum() + 1e-5).cpu().numpy())
        if sample_stage.sum() > 0:
            per_step_vsqf_rewards.append(torch.sum(sample_reward).cpu().numpy() / (sample_stage.sum() + 1e-5).cpu().numpy())
        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        
        if done[0]:
            r_ = np.mean(l_cumu_r.cpu().numpy())            
            dis_r = np.mean(l_cumu_dis_r.cpu().numpy())
            vsqf_r = np.mean(l_cumu_vsqf_r.cpu().numpy())
            
            l_episode_rewards.append(r_)
            l_episode_dis_r.append(dis_r)
            l_episode_vsqf_r.append(vsqf_r)
            print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}, dis reward={dis_r}, vsqf reward={vsqf_r}")
            logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_},  dis reward={dis_r}, vsqf reward={vsqf_r}")

            l_reward = torch.zeros(num_scenes).to(device)
            last_scores = l_reward
            l_cumu_r = torch.zeros(num_scenes).to(device)
            l_cumu_vsqf_r = torch.zeros(num_scenes).to(device)
            l_cumu_dis_r = torch.zeros(num_scenes).to(device)
            
            if args.eval:
                for e, x in enumerate(done):    # if done, maps from new obs
                    if x:
                        episode_done[e].append(True)
                        if len(episode_done[e]) == num_episodes:
                            finished[e] = 1

            if args.agent == "frontier":
                l_policy.reset(num_scenes)
                
        # Sample next action
        if args.agent == "rl":
            l_value, l_action, l_action_log_prob, l_rec_states = \
                l_policy.act(
                    l_rollouts.obs[l_step + 1],
                    l_rollouts.rec_states[l_step + 1],
                    l_rollouts.masks[l_step + 1],
                    extras=l_rollouts.extras[l_step + 1],
                    deterministic=False
                )
            l_action = l_action.cpu().numpy()
        elif args.agent == "random":
            l_action = np.random.randint(0, 3, num_scenes)

        full_map = maps.full_map
        vis_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(vis_inputs):
                
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]

            p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]
            
            # for pointnav
            p_input['time'] = infos[e]['time']
            p_input['depth'] = infos[e]['depth']
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                    ].argmax(0).cpu().numpy()
                full_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()  
                p_input['vsqf_map'] = local_vsqf_map[e, :, :, :].cpu().numpy()
                p_input['vsqf_map_full'] = full_vsqf_map[e, :, :, :].cpu().numpy()                  
        
        if args.agent == "frontier":  # must be after updating vis_inputs
            l_action, goals, short_time_goals = l_policy.get_actions(vis_inputs)        
            if args.visualize or args.print_images:
                for e, p_input in enumerate(vis_inputs):
                    p_input["frontier_goal"] = goals[e]
                    p_input["short_time_goal"] = short_time_goals[e]
        
        # transition: next state
        # pred instance, get semantic masks and step env
        obs, distance_rewards, done, infos = envs.step_and_preprocess(l_action, vis_inputs)    # if done ,envs.reset, obs are ones after reset
        l_action = torch.tensor(l_action)
        # if episode over, reset maps
        for e, x in enumerate(done):    # if done, maps from new obs
            if x:
                maps._init_map_and_pose_for_env(e)
                vsqf_maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
                
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
        
        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if l_step == args.num_local_steps - 1:
            if not args.eval:
                l_next_value = l_policy.get_value(
                    l_rollouts.obs[-1],
                    l_rollouts.rec_states[-1],
                    l_rollouts.masks[-1],
                    extras=l_rollouts.extras[-1]
                ).detach()
                l_rollouts.compute_returns(l_next_value, args.use_gae,
                                           args.gamma, args.tau)
                l_value_loss, l_action_loss, l_dist_entropy = \
                    l_agent.update(l_rollouts)
                l_value_losses.append(l_value_loss)
                l_action_losses.append(l_action_loss)
                l_dist_entropies.append(l_dist_entropy)
            if args.agent == "rl":
                l_rollouts.after_update()       # rollout的最后一个state是下一次initial state
            elif args.agent == "random":
                pass
        torch.set_grad_enabled(False)

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

            log += "\n\tRewards:"

            if len(per_step_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_rewards),
                        np.median(per_step_rewards),
                        np.min(per_step_rewards),
                        np.max(per_step_rewards))
                ])
                
            log += "\n\tDistance Rewards:"

            if len(per_step_dis_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, dis rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_dis_rewards),
                        np.median(per_step_dis_rewards),
                        np.min(per_step_dis_rewards),
                        np.max(per_step_dis_rewards))
                ])
                
            log += "\n\tSample Rewards:"

            if len(per_step_vsqf_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, Vsqf rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_vsqf_rewards),
                        np.median(per_step_vsqf_rewards),
                        np.min(per_step_vsqf_rewards),
                        np.max(per_step_vsqf_rewards))
                ])

            log += "\n\tLosses:"
            if len(l_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(l_value_losses),
                        np.mean(l_action_losses),
                        np.mean(l_dist_entropies))
                ])
                
            if done[0]:
                if len(l_episode_rewards) > 0:
                    log += " ".join([
                    " episode mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(l_episode_rewards),
                        np.median(l_episode_rewards),
                        np.min(l_episode_rewards),
                        np.max(l_episode_rewards))
                    ])
            
            if done[0]:
                if len(l_episode_dis_r) > 0:
                    log += " ".join([
                    " episode mean/med/min/max, dis rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(l_episode_dis_r),
                        np.median(l_episode_dis_r),
                        np.min(l_episode_dis_r),
                        np.max(l_episode_dis_r))
                    ])
            
            if done[0]:
                if len(l_episode_vsqf_r) > 0:
                    log += " ".join([
                    " episode mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(l_episode_vsqf_r),
                        np.median(l_episode_vsqf_r),
                        np.min(l_episode_vsqf_r),
                        np.max(l_episode_vsqf_r))
                    ])
                        
            print(log)
            logging.info(log)


        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(l_episode_rewards) >= 20 and \
                    (np.mean(l_episode_rewards) >= best_l_reward) \
                    and not args.eval:
                torch.save(l_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_l_reward = np.mean(l_episode_rewards)
        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(l_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
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