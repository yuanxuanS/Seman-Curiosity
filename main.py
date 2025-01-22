
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
from  src.policy_rl import algo 
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
    
    device = args.device = torch.device("cuda:1" if args.cuda else "cpu")   # 训练的gpu

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
    per_step_l_rewards = deque(maxlen=1000)
    per_step_rewards = deque(maxlen=1000)
    
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
    
    if args.agent == "rl":
        # Local policy observation space
        es = 3      # extra size: x, y, orientation
        l_observation_space = envs.get_obs_space()[0]  # TODO: VectorEnv's func
        l_action_space = envs.get_action_space()[0]

        # local policy recurrent layer size
        l_hidden_size = args.local_hidden_size

        # Local policy: TODO
        l_policy = RL_Policy(l_observation_space.shape, l_action_space,
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
    
        # Get local policy input
        local_input = obs[:, :3, ...]
        local_orientation = torch.zeros(num_scenes, 1).long()
        local_xy = torch.zeros(num_scenes, 2)
        
        # locs = local_pose.cpu().numpy()
        locs = full_pose.cpu().numpy()      # 使用全局pose
        for e in range(num_scenes):
            local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            local_xy[e] = torch.from_numpy(locs[e, :2][np.newaxis, :])
            
        extras = torch.zeros(num_scenes, es)
        # extras[:, 0] = local_orientation[:, 0]
        extras[:, 2] = local_orientation[:, 0]
        extras[:, :2] = local_xy[:]

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
        l_action = np.random.randint(0, 4, num_scenes)
    
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
    # transition:
    # pred instance, get semantic masks and step env: 
    obs, _, done, infos = envs.step_and_preprocess(l_action, vis_inputs)
    l_action = torch.tensor(l_action)
    # update map
    local_map, local_pose = maps.update_semantic_map(obs, infos)
    full_pose = maps.full_pose
    
    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    logging.info("Start date and time: %s", start_datetime)
    
    l_reward = torch.zeros(num_scenes).to(device)
    last_reward = torch.zeros(num_scenes).to(device)

    
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
            l_reward = last_reward
        else:
            l_reward = args.reward_coeff* maps.sum_of_semantic_map()

        # per step reward? TODO
        # add explore metric: TODO

        # ------------------------------------------------------------------ 
        # update local input, next state
        locs = full_pose.cpu().numpy()
        
        if args.agent == "rl":
            for e in range(num_scenes):
                local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)   # 
                local_xy[e] = torch.from_numpy(locs[e, :2][np.newaxis, :])
                
            local_input = obs[:, :3, ...]       # rgb
            extras[:, 0] = local_orientation[:, 0]
            extras[:, :2] = local_xy[:]
            # print(f"input sxtras: {extras}")
        # Add samples to local policy storage
        reward = l_reward - last_reward
        
        if args.agent == "rl":
            l_rollouts.insert(
                    local_input, l_rec_states,      # state_t+1
                    l_action, l_action_log_prob, l_value,   # action, reward_t
                    reward, l_masks, extras
                )
        last_reward = l_reward

        # 
        reward_mean = np.mean(reward.cpu().numpy())
        l_reward_mean = np.mean(l_reward.cpu().numpy())
        per_step_rewards.append(reward_mean)
        per_step_l_rewards.append(l_reward_mean)

        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        
        if done[0]:
            r_ = np.mean(l_reward.cpu().numpy())
            print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
            logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
            l_episode_rewards.append(r_)

            l_reward = torch.zeros(num_scenes).to(device)
            last_reward = l_reward
            
            if args.eval:
                for e, x in enumerate(done):    # if done, maps from new obs
                    if x:
                        episode_done[e].append(True)
                        if len(episode_done[e]) == num_episodes:
                            finished[e] = 1

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
        # print(f"action {l_action}")
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
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                    ].argmax(0).cpu().numpy()
                full_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred_full'] = full_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()

        # transition: next state
        # pred instance, get semantic masks and step env
        obs, _, done, infos = envs.step_and_preprocess(l_action, vis_inputs)    # if done ,envs.reset, obs are ones after reset
        l_action = torch.tensor(l_action)
        # if episode over, reset maps
        for e, x in enumerate(done):    # if done, maps from new obs
            if x:
                maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
                
        # update map
        local_map, local_pose = maps.update_semantic_map(obs, infos)
        full_pose = maps.full_pose
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