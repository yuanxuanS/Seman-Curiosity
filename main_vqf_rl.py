
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
from src.policy_rl.model import RL_Policy
from src.policy_rl.model_vqf import RL_VQF_Policy
from  src.policy_rl import algo 
from src.policy_rl.baseline_frontier import Frontier
import cv2
import json
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
    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        # TODO: 增加一些online指标
        episode_done = []
        for _ in range(args.num_processes):
            episode_done.append(deque(maxlen=num_episodes))
    else:
        pass # TODO: 
    
    finished = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)  # 每个环境的每个episode的奖励
    per_step_g_rewards = deque(maxlen=1000)
    per_step_rewards = deque(maxlen=1000)
    
    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      
    obs, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?
    tgt_class = [info['target_cls'] for info in infos]
    # vqf_stage
    
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


    if args.agent == "rl":
        # Global policy observation space
        es = 0      # extra size
        ngc = 4 + 1
        g_observation_space = gym.spaces.Box(0, 1,      
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')
        g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

        # global policy recurrent layer size
        g_hidden_size = args.global_hidden_size

        # Global policy: TODO
        g_policy = RL_VQF_Policy(g_observation_space.shape, g_action_space,
                            model_type=1,
                            base_kwargs={'recurrent': args.use_recurrent_global,
                                        'hidden_size': g_hidden_size,
                                        'num_sem_categories': args.num_sem_categories
                                        }).to(device)
        
        g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                        args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)

        

        # Storage: 
        g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                        num_scenes, g_observation_space.shape,
                                        g_action_space, g_policy.rec_state_size,
                                        es).to(device)
        
        # load weights
        if args.load != "0":
            print("Loading model {}".format(args.load))
            logging.info("Loading model {}".format(args.load))
            state_dict = torch.load(args.load,
                                    map_location=lambda storage, loc: storage)
            g_policy.load_state_dict(state_dict)

        if args.eval:
            g_policy.eval()
    
        # Get local policy input
        global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
        
        global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()  # local_map的 obstacle, explored, curr loc, past loc
        # global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        #     full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
        for e in range(num_scenes):
            if tgt_class[e] != None:
                global_input[:, 4, :, :] = local_map[:, 4+tgt_class[e], :, :].detach()     # target local semantic maps

        extras = torch.zeros(num_scenes, es)
        
        # locs = local_pose.cpu().numpy()
        g_rollouts.obs[0].copy_(global_input)   # 

        # Run Global policy
        g_value, g_action, g_action_log_prob, g_rec_states = \
            g_policy.act(
                g_rollouts.obs[0],
                g_rollouts.rec_states[0],
                g_rollouts.masks[0],
                extras=g_rollouts.extras[0],
                deterministic=False
            )
        cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
        global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
        global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
        
        for e, p_input in enumerate(vis_inputs):
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = 1
    
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
    elif args.agent == "frontier_rl":
        
        policy = ["vqf_rl" if tgt_class[e] != None else "frontier" for e in range(num_scenes)]     # "vqf_rl"
        
        frontier_policy = Frontier(args)
        frontier_policy.reset(num_scenes)
        l_action, goals, short_time_goals = frontier_policy.get_actions(vis_inputs)        
        for e, p_input in enumerate(vis_inputs):
            if args.visualize or args.print_images:
                p_input["frontier_goal"] = goals[e]
                p_input["short_time_goal"] = short_time_goals[e]
        
        
        l_observation_space = envs.get_obs_space()[0]  # TODO: VectorEnv's func
        l_action_space = envs.get_action_space()[0]
        l_policy_rl = RL_VQF_Policy(l_observation_space.shape, l_action_space,
                            model_type=1,
                            base_kwargs={'recurrent': args.use_recurrent_local,
                                        'hidden_size': l_hidden_size,
                                        'num_sem_categories': args.num_sem_categories
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
            
            
    ## transition:
    # pred instance, get semantic masks and step env: 
    obs, _, done, infos = envs.plan_act_and_preprocess(vis_inputs)
    tgt_class = [info['target_cls'] for info in infos]
    
    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    logging.info("Start date and time: %s", start_datetime)
    
    g_reward = torch.zeros(num_scenes).to(device)

    
    torch.set_grad_enabled(False)

    print("Starting running")
    logging.info("Starting running")
    if not args.eval:
        print(f"training frames is {args.num_training_frames}")
        logging.info(f"training frames is {args.num_training_frames}")
    for step in range(args.num_training_frames // args.num_processes + 1):
        l_step = step % args.num_local_steps
        g_step = (step // args.num_local_steps) % args.num_global_steps
        
        if finished.sum() == args.num_processes:    # eval over
            break
        
        # get reward: TODO
        # if done[0]:     # maps are new obs, sum of map will be small, and get negative reward
        #     l_reward = last_reward
        # else:
        #     l_reward = args.reward_coeff* maps.sum_of_semantic_map()

        # per step reward? TODO
        # add explore metric: TODO

        # For every global step, update the full and local mapss
        local_map, local_pose = maps.update_local_semantic_map(obs, infos)
        maps.local_map, maps.local_pose = local_map, local_pose
        
        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # ------------------------------------------------------------------ 
            # update global input, next state
            local_map, local_pose = maps.update_full_semantic_map(local_map, local_pose)
            full_pose = maps.full_pose
            full_map = maps.full_map
            
            if args.agent == "rl":
                locs = local_pose.cpu().numpy()    
                
                global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()  # local_map的 obstacle, explored, curr loc, past loc
                # global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
                #     full_map[:, 0:4, :, :])     # full_map的 obstacle, explored, curr loc, past loc
                for e in range(num_scenes):
                    if tgt_class[e] != None:
                        global_input[:, 4, :, :] = local_map[:, 4+tgt_class[e], :, :].detach()     # local semantic maps

            # update global reward: TODO, vqf increase of all steps
            g_reward = torch.tensor([infos[env_idx]['g_reward'] / args.num_local_steps \
                        for env_idx in range(num_scenes)]).to(device)

            if args.agent == "rl":
                g_rollouts.insert(
                        global_input, g_rec_states,      # state_t+1
                        g_action, g_action_log_prob, g_value,   # action, reward_t
                        g_reward, g_masks, extras
                    )

        # 
                # reward_mean = np.mean(g_reward.cpu().numpy())       # TODO
                # l_reward_mean = np.mean(l_reward.cpu().numpy())
                # per_step_rewards.append(reward_mean)
                # per_step_l_rewards.append(l_reward_mean)

        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        
            if done[0]:
                r_ = np.mean(g_reward.cpu().numpy())
                print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
                logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
                g_episode_rewards.append(r_)

                g_reward = torch.zeros(num_scenes).to(device)
            
            # Sample next action
            if args.agent == "rl":
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                global_goals = [[int(action[0] * local_w),
                                int(action[1] * local_h)]
                                for action in cpu_actions]
                global_goals = [[min(x, int(local_w - 1)),
                                min(y, int(local_h - 1))]
                                for x, y in global_goals]
                g_reward = 0
            elif args.agent == "random":
                l_action = np.random.randint(0, 3, num_scenes)
            elif args.agent == "frontier_rl":
                pass
            
        if args.eval:
            for e, x in enumerate(done):    # if done, maps from new obs
                if x:
                    episode_done[e].append(True)
                    if len(episode_done[e]) == num_episodes:
                        finished[e] = 1

            if args.agent == "frontier":
                frontier_policy.reset(num_scenes)
                
            
        
        if step%500 == 250:
            maps.filter_obstacle_map()
            for e, p_input in enumerate(vis_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()                
                p_input['map_pred_full'] = full_map[e, 0, :, :].cpu().numpy()
            envs.update_collision_map(vis_inputs)
        
        # ------------------------------------------------------------------
        # action: update subgoal 
        # TODO: Update long-term goal if target object is found
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
            
        full_map = maps.full_map
        vis_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(vis_inputs):
                
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = maps.get_all_pose()[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
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
        
        if args.agent == "frontier":  # must be after updating vis_inputs
            l_action, goals, short_time_goals = l_policy.get_actions(vis_inputs)        
            if args.visualize or args.print_images:
                for e, p_input in enumerate(vis_inputs):
                    p_input["frontier_goal"] = goals[e]
                    p_input["short_time_goal"] = short_time_goals[e]
        
        # transition: next state
        # pred instance, get semantic masks and step env
        obs, _, done, infos = envs.plan_act_and_preprocess(vis_inputs)    # if done ,envs.reset, obs are ones after reset

        # if episode over, reset maps
        for e, x in enumerate(done):    # if done, maps from new obs
            if x:
                maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
                
        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
            and l_step == args.num_local_steps - 1:
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=l_rollouts.extras[-1]
                ).detach()
                g_rollouts.compute_returns(g_next_value, args.use_gae,
                                           args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            if args.agent == "rl":
                g_rollouts.after_update()       # rollout的最后一个state是下一次initial state
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

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_value_losses),
                        np.mean(g_action_losses),
                        np.mean(g_dist_entropies))
                ])
                
            if done[0]:
                if len(g_episode_rewards) > 0:
                    log += " ".join([
                    " episode mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                    ])
                
            print(log)
            logging.info(log)


        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 20 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)
        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                if args.agent == "rl":
                    torch.save(g_policy.state_dict(),
                            os.path.join(dump_dir,
                                            "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------
    # Print and save model performance numbers during evaluation: TODO
    # with open('{}/{}_episode_rewards.json'.format(
    #         dump_dir, args.split), 'w') as f:
    #     json.dump(l_episode_rewards, f)
    m = np.array(g_episode_rewards).mean()
    print(f"all episode rewards: {g_episode_rewards}, mean is {m}")
    logging.info(f"all episode rewards: {g_episode_rewards}, mean is {m}")
    np.savez('{}/{}_episode_rewards.npz'.format(
            dump_dir, args.split), episode_reward=g_episode_rewards)
    
    if args.eval:
        print("Dumping eval details...")
        

        
if __name__ == "__main__":
    main()