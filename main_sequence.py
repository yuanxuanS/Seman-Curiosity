
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
from src.policy_rl.utils.storage import GlobalRolloutStorage
from src.policy_rl.model import RL_Policy, RL_Policy2
from src.policy_rl.expert_predictor import ExpertPredictor
from  src.policy_rl import algo 
from src.policy_rl.baseline_frontier import Frontier
import cv2
import json

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
    
    device = args.device = torch.device("cuda:3" if args.cuda else "cpu")   # 训练的gpu

    #  l_masks, not used. episode length不同时使用
    l_masks = torch.ones(num_scenes).float().to(device)

    best_reward = -np.inf

    if args.eval:
        # TODO: 增加一些online指标
        episode_done = []
        for _ in range(args.num_processes):
            episode_done.append(deque(maxlen=num_episodes))
    else:
        pass # TODO: 
    
    finished = np.zeros((args.num_processes))

    episode_rewards = []
    per_step_incre_rewards = deque(maxlen=1000)
    per_step_curr_rewards = deque(maxlen=1000)
    
    value_losses = deque(maxlen=1000)
    action_losses = deque(maxlen=1000)
    dist_entropies = deque(maxlen=1000)
    distill_losses = deque(maxlen=1000)

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      
    obs_all, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?

    torch.set_grad_enabled(False)

    # Initializing Maps
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    maps = Maps_Env(args)
    obs = torch.concat(obs_all, axis=0)
    sensor_pose = [infos[e]['sensor_pose_all'][0] for e in range(num_scenes)]
    orients = [infos[e]['orient_idx'] for e in range(num_scenes)]
    straight = [False for e in range(num_scenes)]
    local_map, local_pose = maps.update_semantic_map2(obs, sensor_pose, orients, straight)
    full_pose = maps.full_pose
    
    # for visualize
    
    full_map = maps.full_map
    vis_inputs = [{} for e in range(num_scenes)]
    orient_full_map = maps.orient_full_maps
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
            
            p_input['orient_full_map_0_obsta'] = orient_full_map[0][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_0_exp'] = orient_full_map[0][e, 1, :, :].cpu().numpy()
            p_input['orient_full_map_1_obsta'] = orient_full_map[1][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_1_exp'] = orient_full_map[1][e, 1, :, :].cpu().numpy()
            p_input['orient_full_map_2_obsta'] = orient_full_map[2][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_2_exp'] = orient_full_map[2][e, 1, :, :].cpu().numpy()
            p_input['orient_full_map_3_obsta'] = orient_full_map[3][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_3_exp'] = orient_full_map[3][e, 1, :, :].cpu().numpy()
            p_input['orient_full_map_4_obsta'] = orient_full_map[4][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_4_exp'] = orient_full_map[4][e, 1, :, :].cpu().numpy()
            p_input['orient_full_map_5_obsta'] = orient_full_map[5][e, 0, :, :].cpu().numpy()
            p_input['orient_full_map_5_exp'] = orient_full_map[5][e, 1, :, :].cpu().numpy()
                     
    vis_inputs_frames = [vis_inputs]

    if args.agent == "rl":
        # Local policy observation space
        es = 0      # extra size: x, y, orientation
        observation_space = envs.get_obs_space()[0]  # TODO: VectorEnv's func
        action_space = envs.get_action_space()[0]

        # local policy recurrent layer size
        hidden_size = args.local_hidden_size

        # Local policy: TODO
        policy = RL_Policy2(observation_space.shape, action_space,
                            device = device,
                            # model_type=1,
                            # base_kwargs={'recurrent': args.use_recurrent_local,
                            #             'hidden_size': l_hidden_size,
                            #             'num_sem_categories': args.num_sem_categories
                            #             }
                            ).to(device)
        
        agent = algo.PPO(policy, args.clip_param, args.ppo_epoch,
                        args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)
        
        # Initialize Expert Predictor for knowledge distillation
        expert_predictor = ExpertPredictor(device="cuda:3")

        
    
        # Storage: 
        rollouts = GlobalRolloutStorage(args.num_local_steps,
        # rollouts = GlobalRolloutStorage(50,
                                        num_scenes, observation_space.shape,
                                        action_space, 1,
                                        es).to(device)
        
        # load weights
        if args.load != "0":
            print("Loading model {}".format(args.load))
            logging.info("Loading model {}".format(args.load))
            state_dict = torch.load(args.load,
                                    map_location=lambda storage, loc: storage)
            policy.load_state_dict(state_dict)

        if args.eval:
            policy.eval()
    
        # Get local policy input
        # local_input = np.concatenate((obs[:, :3, ...], obs[:, 4, ...][:, np.newaxis, ...]), axis=1)
        # local_input = obs[:, :3, ...]
        local_input = [torch.from_numpy(info['panorama_obs']).unsqueeze(0) for info in infos]
        local_input = torch.concat(local_input, axis=0)
        rec_states = torch.zeros( num_scenes,
                                      1)
        # local_orientation = torch.zeros(num_scenes, 1).long()
        # local_xy = torch.zeros(num_scenes, 2)
        
        # locs = local_pose.cpu().numpy()
        # locs = full_pose.cpu().numpy()      # 使用全局pose
        # for e in range(num_scenes):
        #     local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
        #     local_xy[e] = torch.from_numpy(locs[e, :2][np.newaxis, :])
            
        extras = torch.zeros(num_scenes, es)
        # extras[:, 0] = local_orientation[:, 0]
        # extras[:, 2] = local_orientation[:, 0]
        # extras[:, :2] = local_xy[:]

        rollouts.obs[0].copy_(local_input)   # 
        rollouts.extras[0].copy_(extras)

        # Run Local policy
        value, action, action_log_prob = \
            policy.act(
                rollouts.obs[0],
                # rollouts.rec_states[0],
                # rollouts.masks[0],
                extras=rollouts.extras[0],
                deterministic=False
            )
        action = action.cpu().numpy()
    
    elif args.agent == "random":
        action = np.random.randint(0, 12, num_scenes)
    elif args.agent == "frontier":
        policy = Frontier(args)
        policy.reset(num_scenes)
        action, goals, short_time_goals = policy.get_actions(vis_inputs)        
        for e, p_input in enumerate(vis_inputs):
            if args.visualize or args.print_images:
                p_input["frontier_goal"] = goals[e]
                p_input["short_time_goal"] = short_time_goals[e]
    # transition:
    # pred instance, get semantic masks and step env: 
    actions = []
    actions.append(action)
    # print(f"action is {l_action}")
    obs_all, _, done, infos = envs.step_and_preprocess(action, vis_inputs_frames)
    action = torch.tensor(action)
        
    max_len = max([len(obs) for obs in obs_all])
    obs_pad_all = []
    obs_paded = torch.zeros_like(obs_all[0][0]).to(obs_all[0][0].device)
    straight = [[] for e in range(num_scenes)]
    for e, obs_ in enumerate(obs_all): 
        straight_ = infos[e]['no_straight_num']*[False]
        straight_.extend([True]*(len(obs_) - infos[e]['no_straight_num']))
        if len(obs_) == max_len:
            obs_pad_all.append(torch.concat(obs_, axis=0).unsqueeze(0))
            straight[e] = straight_
            continue
        infos[e]['sensor_pose_all'].extend([[0.,0.,0.]] * (max_len - len(obs_)))
        infos[e]['orient_idx'].extend([0] * (max_len - len(obs_)))
        
        straight_.extend([False]*(max_len - len(obs_)))
        straight[e] = straight_
        obs_.extend([obs_paded]*(max_len - len(obs_)))
        obs_pad_all.append(torch.concat(obs_, axis=0).unsqueeze(0))
        
    # update map
    obs_pad_all = torch.concat(obs_pad_all, axis=0)
    vis_inputs_frames = []
    for frame in range(obs_pad_all.shape[1]):
        obs = obs_pad_all[:, frame, ...]
        sensor_pose = [infos[e]['sensor_pose_all'][frame] for e in range(num_scenes)]
        orients = [infos[e]['orient_idx'][frame] for e in range(num_scenes)]
        stra_ =  [straight[e][frame] if len(straight[e]) > 0 else False for e in range(num_scenes)]
        local_map, local_pose = maps.update_semantic_map2(obs, sensor_pose, orients, stra_)

        # for visualzie 
        full_map = maps.full_map
        vis_inputs_ = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(vis_inputs_):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]

            p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]
            
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                    ].argmax(0).cpu().numpy()
                full_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()                    
        vis_inputs_frames.append(vis_inputs_)
            
            
    full_pose = maps.full_pose
    
    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    logging.info("Start date and time: %s", start_datetime)
    
    l_reward = torch.zeros(num_scenes).to(device)
    step_other_reward = torch.zeros(num_scenes).to(device)
    last_reward = torch.zeros(num_scenes).to(device)

    torch.set_grad_enabled(False)

    print("Starting running")
    logging.info("Starting running")
    if not args.eval:
        print(f"training frames is {args.num_training_frames}")
        logging.info(f"training frames is {args.num_training_frames}")
    for step in range(args.num_training_frames // args.num_processes + 1):
        l_step = step % args.num_local_steps
        # l_step = step % 100
        
        if finished.sum() == args.num_processes:    # eval over
            break
                
        # get reward: map change after state transition
        if done[0]:     # maps are new obs, sum of map will be small, and get negative reward
            l_reward = last_reward
        else:
            sequence_r =  torch.tensor([infos[e]['sequence_reward'] for e in range(num_scenes)]).to(last_reward.device)
            cls_entropy_r =   torch.tensor([infos[e]['cls_etp'] for e in range(num_scenes)]).to(last_reward.device)
            # l_reward = last_reward + sequence_reward  #  + cls_entropy_r
            # l_reward = args.reward_coeff* 30 *maps.sum_of_orient_semantic_map()
            # l_reward = args.reward_coeff* maps.sum_of_explore_map()
            # l_reward = args.reward_coeff* maps.sum_of_orient_map()
            l_reward = args.reward_coeff* maps.sum_of_semantic_map()
            step_other_reward += cls_entropy_r  + sequence_r
            
        # divesity reward
        # if args.use_diversity_reward:
        #     l_reward += diversity_reward

        # ------------------------------------------------------------------ 
        # update local input, next state
        locs = full_pose.cpu().numpy()
        
        if args.agent == "rl":
            # for e in range(num_scenes):
            #     local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)   # 
            #     local_xy[e] = torch.from_numpy(locs[e, :2][np.newaxis, :])
                
            local_input = [torch.from_numpy(info['panorama_obs']).unsqueeze(0) for info in infos]
            local_input = torch.concat(local_input, axis=0)
            # extras[:, 0] = local_orientation[:, 0]
            # extras[:, :2] = local_xy[:]
            # print(f"input sxtras: {extras}")
        # Add samples to local policy storage
        reward = l_reward - last_reward + cls_entropy_r + sequence_r
        
        if args.agent == "rl":
            # Get expert_probs from ExpertPredictor using panorama_obs_all (batch inference)
            panorama_obs_list = [infos[i]['panorama_obs_all'] for i in range(num_scenes)]
            expert_probs_batch = expert_predictor.predict(panorama_obs_list)
            
            rollouts.insert(
                    local_input, rec_states,      # state_t+1
                    action, action_log_prob, value,   # action, reward_t
                    reward, l_masks, extras,
                    expert_probs=expert_probs_batch
                )
        last_reward = l_reward

        # 
        reward_incre_mean = np.mean(reward.cpu().numpy())
        reward_curr_mean = np.mean(l_reward.cpu().numpy())
        per_step_incre_rewards.append(reward_incre_mean)
        per_step_curr_rewards.append(reward_curr_mean)

        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        
        if done[0]:
            r_ = np.mean(l_reward.cpu().numpy())
            r_so = np.mean(step_other_reward.cpu().numpy())
            print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode map reward={r_}\nepisode sequence reward={r_so}\nepisode mean reward={r_+r_so}")
            logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode map reward={r_}\nepisode sequence reward={r_so}\nepisode mean reward={r_+r_so}")
            episode_rewards.append(r_)

            l_reward = torch.zeros(num_scenes).to(device)
            step_other_reward = torch.zeros(num_scenes).to(device)
            last_reward = l_reward
            
            if args.eval:
                for e, x in enumerate(done):    # if done, maps from new obs
                    if x:
                        episode_done[e].append(True)
                        if len(episode_done[e]) == num_episodes:
                            finished[e] = 1

            if args.agent == "frontier":
                policy.reset(num_scenes)
                
        # Sample next action
        if args.agent == "rl":
            value, action, action_log_prob = \
                policy.act(
                    rollouts.obs[l_step + 1],
                    # rollouts.rec_states[l_step + 1],
                    # rollouts.masks[l_step + 1],
                    # extras=rollouts.extras[l_step + 1],
                    deterministic=False
                )
            action = action.cpu().numpy()
        elif args.agent == "random":
            action = np.random.randint(0, 12, num_scenes)

        # full_map = maps.full_map
        # vis_inputs = [{} for e in range(num_scenes)]
        # for e, p_input in enumerate(vis_inputs):
                
        #     p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        #     p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        #     p_input['pose_pred'] = maps.get_all_pose()[e]

        #     p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
        #     p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
        #     p_input['pose_pred'] = maps.get_all_pose()[e]
            

        #     if args.visualize or args.print_images:
        #         local_map[e, -1, :, :] = 1e-5
        #         p_input['sem_map_pred'] = local_map[e, 4:, :, :
        #                                             ].argmax(0).cpu().numpy()
        #         full_map[e, -1, :, :] = 1e-5
        #         p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
        #                                                 ].argmax(0).cpu().numpy()                    
        
        if args.agent == "frontier":  # must be after updating vis_inputs
            action, goals, short_time_goals = policy.get_actions(vis_inputs)        
            if args.visualize or args.print_images:
                for e, p_input in enumerate(vis_inputs):
                    p_input["frontier_goal"] = goals[e]
                    p_input["short_time_goal"] = short_time_goals[e]
        
        
        # transition: next state
        # pred instance, get semantic masks and step env
        actions.append(action)
        # print(f"action is {l_action}")
        obs_all, _, done, infos = envs.step_and_preprocess(action, vis_inputs_frames)    # if done ,envs.reset, obs are ones after reset
        action = torch.tensor(action)
        # if episode over, reset maps
        for e, x in enumerate(done):    # if done, maps from new obs
            if x:
                maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
        
        
        # update map
        max_len = max([len(obs) for obs in obs_all])
        obs_pad_all = []
        obs_paded = torch.zeros_like(obs_all[0][0]).to(obs_all[0][0].device)
        straight = [[] for e in range(num_scenes)]
        for e, obs_ in enumerate(obs_all): 
            straight_= [False] * infos[e]['no_straight_num']
            straight_.extend([True]*(len(obs_) - infos[e]['no_straight_num']))
            if len(obs_) == max_len:
                obs_pad_all.append(torch.concat(obs_, axis=0).unsqueeze(0))
                straight[e] = straight_
                continue
            infos[e]['sensor_pose_all'].extend([[0.,0.,0.]] * (max_len - len(obs_)))
            infos[e]['orient_idx'].extend([0] * (max_len - len(obs_)))
            
            straight_.extend([False]*(max_len - len(obs_)))
            straight[e] = straight_
            obs_.extend([obs_paded]*(max_len - len(obs_)))
            obs_pad_all.append(torch.concat(obs_, axis=0).unsqueeze(0))
            
        # 将所有环境的frame的观测对其，遍历frame更新地图
        obs_pad_all = torch.concat(obs_pad_all, axis=0)
        vis_inputs_frames = []
        orient_full_map = maps.orient_full_maps
        for frame in range(obs_pad_all.shape[1]):
            obs = obs_pad_all[:, frame, ...]
            sensor_pose = [infos[e]['sensor_pose_all'][frame] for e in range(num_scenes)]
            orients = [infos[e]['orient_idx'][frame] for e in range(num_scenes)]
            stra_ =  [straight[e][frame] if len(straight[e]) > 0 else False for e in range(num_scenes)]
            local_map, local_pose = maps.update_semantic_map2(obs, sensor_pose, orients, stra_)
            
            # for visualzie 
            full_map = maps.full_map
            vis_inputs_ = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(vis_inputs_):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = maps.get_all_pose()[e]

                p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred_full'] = full_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = maps.get_all_pose()[e]
                
                if args.visualize or args.print_images:
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()
                    full_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
                                                            ].argmax(0).cpu().numpy() 
                    p_input['orient_full_map_0_obsta'] = orient_full_map[0][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_0_exp'] = orient_full_map[0][e, 1, :, :].cpu().numpy()
                    p_input['orient_full_map_1_obsta'] = orient_full_map[1][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_1_exp'] = orient_full_map[1][e, 1, :, :].cpu().numpy()
                    p_input['orient_full_map_2_obsta'] = orient_full_map[2][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_2_exp'] = orient_full_map[2][e, 1, :, :].cpu().numpy()
                    p_input['orient_full_map_3_obsta'] = orient_full_map[3][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_3_exp'] = orient_full_map[3][e, 1, :, :].cpu().numpy()
                    p_input['orient_full_map_4_obsta'] = orient_full_map[4][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_4_exp'] = orient_full_map[4][e, 1, :, :].cpu().numpy()
                    p_input['orient_full_map_5_obsta'] = orient_full_map[5][e, 0, :, :].cpu().numpy()
                    p_input['orient_full_map_5_exp'] = orient_full_map[5][e, 1, :, :].cpu().numpy()
                                   
            vis_inputs_frames.append(vis_inputs_)

        full_pose = maps.full_pose
        
        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if l_step == args.num_local_steps - 1:
        # if l_step == 100 - 1:
            if not args.eval and args.agent == "rl":
                next_value = policy.get_value(
                    rollouts.obs[-1],
                    # rollouts.rec_states[-1],
                    # rollouts.masks[-1],
                    extras=None
                ).detach()
                rollouts.compute_returns(next_value, args.use_gae,
                                           args.gamma, args.tau)
                
                value_loss, action_loss, dist_entropy, distill_loss = \
                    agent.update_with_supervise(rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                distill_losses.append(distill_loss)
            if args.agent == "rl":
                rollouts.after_update()       # rollout的最后一个state是下一次initial state
            elif args.agent == "random":
                pass
        torch.set_grad_enabled(False)

        # st = time.time()
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

            if len(per_step_curr_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_curr_rewards),
                        np.median(per_step_curr_rewards),
                        np.min(per_step_curr_rewards),
                        np.max(per_step_curr_rewards))
                ])

            log += "\n\tLosses:"
            if len(value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(value_losses),
                        np.mean(action_losses),
                        np.mean(dist_entropies))
                ])
                log += " Distill Loss: {:.3f},".format(np.mean(distill_losses))
                
            if done[0]:
                if len(episode_rewards) > 0:
                    log += " ".join([
                    " episode mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards))
                    ])
                
            print(log)
            logging.info(log)


        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(episode_rewards) >= 20 and \
                    (np.mean(episode_rewards) >= best_reward) \
                    and not args.eval and args.agent == "rl":
                torch.save(policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_reward = np.mean(episode_rewards)
        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval and args.agent == "rl":
                torch.save(policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------
    # Print and save model performance numbers during evaluation: TODO
    # with open('{}/{}_episode_rewards.json'.format(
    #         dump_dir, args.split), 'w') as f:
    #     json.dump(l_episode_rewards, f)
    m = np.array(episode_rewards).mean()
    print(f"all episode rewards: {episode_rewards}, mean is {m}")
    logging.info(f"all episode rewards: {episode_rewards}, mean is {m}")
    np.savez('{}/{}_episode_rewards.npz'.format(
            dump_dir, args.split), episode_reward=episode_rewards)
    
    np.savez('actions.npz', actions=np.array(actions))
    if args.eval:
        print("Dumping eval details...")
        

        
if __name__ == "__main__":
    main()