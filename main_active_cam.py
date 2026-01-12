
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
from src.policy_rl.utils.storage import GlobalRolloutStorage
import cv2
import json
from src.policy_rl.model import RL_Policy
from  src.policy_rl import algo 
from PIL import Image
import clip
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
    
    # clip model
    print("loading clip")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    print("loading clip done.")
    from src.vqf_constants import target_coco_categories
    
    # text_features = {}
    # for i in target_coco_categories.keys():
    #     text = clip.tokenize([f"a photo contains a {i}." ]).to(device)
    #     text_features[i] = text
    
    text_features = clip.tokenize([f"a photo contains a {goal}." for goal in ['chair','couch', 'bed','toilet', 'refrigerator']]).to(device)
    def clip_score(imgs, scene_goal_idx):
        '''
        img: tensors
        scene_goal_idx: list, target idx of goal, defined in target_coco_categories
        '''
        mapping = {"chair":0, "couch":1, "bed":2, "toilet":3, "refrigerator":4}
        scene_goal_idx_ = [mapping[goal_idx] for goal_idx in scene_goal_idx]
        scores = []
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(imgs, text_features)
            for i in range(num_scenes):
                scores.append(logits_per_image[i, scene_goal_idx_[i]].item())
        return torch.tensor(scores).to(device)
    #  l_masks, not used. episode length不同时使用
    l_masks = torch.ones(num_scenes).float().to(device)

    best_l_reward = -np.inf
    
    if args.eval:
        episode_done = []
        for _ in range(args.num_processes):
            episode_done.append(deque(maxlen=num_episodes))
            
        # for saving obs
        obs_info = init_obs_info = None
    else:
        pass # TODO: 
    
    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))
    
    l_episode_rewards = []
    per_step_l_rewards = deque(maxlen=1000)
    per_step_rewards = deque(maxlen=1000)
    
    l_value_losses = deque(maxlen=1000)
    l_action_losses = deque(maxlen=1000)
    l_dist_entropies = deque(maxlen=1000)
    
    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      
    obs, infos = envs.reset()   
    
    
    # process for clip
    images = []
    for n in range(num_scenes):
        pil_img = Image.fromarray(obs[n, :3, ...].cpu().numpy().transpose(1,2,0).astype(np.uint8))
        image = preprocess(pil_img).to(device)
        images.append(image)
    images = torch.stack(images)
    
    
    last_scores = torch.zeros(num_scenes).to(device)
    init_scores = torch.zeros(num_scenes).to(device)
    # clip score 初始化
    goal_idxs = [info['goal_name'] for info in infos]
    init_scores = last_scores = clip_score(images, goal_idxs)
    
    vis_info = None
    if args.visualize or args.print_images:
        vis_info = {"init score": init_scores, "score": init_scores, 'reward': torch.tensor([0]*num_scenes), 'action':torch.tensor([-1]*num_scenes)}
        envs.visualize(vis_info)
        
    torch.set_grad_enabled(False)
    
    # policy
    if args.agent == "rl":
        l_observation_space = envs.get_obs_space()[0]  # TODO: VectorEnv's func
        l_action_space = envs.get_action_space()[0]
        
        # policy recurrent layer size
        l_hidden_size = args.local_hidden_size
        
        # Local policy
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
                                        0).to(device)

        # load weights
        if args.load != "0":
            print("Loading model {}".format(args.load))
            logging.info("Loading model {}".format(args.load))
            state_dict = torch.load(args.load,
                                    map_location=lambda storage, loc: storage)
            l_policy.load_state_dict(state_dict)
        
        if args.eval:
            l_policy.eval()
        
        # Get policy input
        l_input = obs[:, :3, ...]
        
        extras = torch.zeros(num_scenes, 0)
        
        l_rollouts.obs[0].copy_(l_input)   # 

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
        l_action = np.random.randint(0, 5, num_scenes)
    
    if args.eval:
        obs_info = init_obs_info = obs_info_ = envs.get_obs_info()
        capture = [False]*num_scenes
        for e in range(num_scenes):
            if l_action[e] == 0:
                capture[e] = True
        envs.save_data(obs_info, capture)
    
    # transition:
    # pred instance, get semantic masks and step env: 
    obs, _, done, infos = envs.step_and_wait(l_action, wait_env)
    l_action = torch.tensor(l_action)
    
    
    
    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    logging.info("Start date and time: %s", start_datetime)
    
    l_reward = torch.zeros(num_scenes).to(device)
    
    
    cumulative_reward = torch.zeros(num_scenes).to(device)
    
    torch.set_grad_enabled(False)
    
    print("Starting running")
    logging.info("Starting running")
    if not args.eval:
        print(f"training frames is {args.num_training_frames}")
        logging.info(f"training frames is {args.num_training_frames}")
    for step in range(args.num_training_frames // args.num_processes + 1):
        l_step = step % args.num_local_steps
        
        
            
        

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)     # for insert 存在不同时done
        
        ## get reward
        # process for clip
        images = []
        for n in range(num_scenes):
            pil_img = Image.fromarray(obs[n, :3, ...].cpu().numpy().transpose(1,2,0).astype(np.uint8))
            image = preprocess(pil_img).to(device)
            images.append(image)
        images = torch.stack(images)
                
        l_scores = clip_score(images, goal_idxs)
        l_reward = last_scores - l_scores
        # penalty when lost object
        lost_goal = np.array([info['lost_goal'] for info in infos])
        l_reward = torch.where(torch.from_numpy(lost_goal).to(device), -5, l_reward)
        # penalty -10 when lost object
        final_reward = torch.where(init_scores-l_scores>0, 5 + init_scores- l_scores , init_scores- l_scores)
        final_reward = torch.where(torch.from_numpy(lost_goal).to(device), -10, final_reward)
        
        l_reward = torch.where(torch.from_numpy(done).to(device), final_reward, l_reward)
        l_reward = torch.where(torch.from_numpy(wait_env.astype(bool)).to(device), torch.zeros_like(l_reward).to(device), l_reward)
        
        
        if finished.sum() == args.num_processes:    # eval over
            break
        

        # update local input with next state
        if args.agent == "rl":
            l_input = obs[:, :3, ...]       # rgb
        
        # Add samples to local policy storage
        if args.agent == "rl":
            l_rollouts.insert(
                    l_input, l_rec_states,      # state_t+1
                    l_action, l_action_log_prob, l_value,   # action, reward_t
                    l_reward, l_masks, extras
                )
        
        # if lost goal, last score remain for compare score when find goal again
        last_scores = torch.where(torch.from_numpy(lost_goal).to(device), last_scores, l_scores)
        
        if args.visualize or args.print_images:
            vis_info = {'init score': init_scores, 'score':l_scores, 'reward':l_reward, 'action':l_action}
            envs.visualize(vis_info)
            
            
        # record
        cumulative_reward += l_reward
        if wait_env.sum() < num_scenes:
            l_reward_mean = np.sum(l_reward.cpu().numpy()) / (~wait_env.astype(bool)).sum()
        else:
            l_reward_mean = 0
        per_step_l_rewards.append(l_reward_mean)
        
        for e, x in enumerate(done):
            wait_env[e] = 1 if x else wait_env[e]
            
        if l_step == args.num_local_steps - 1:
            r_ = np.mean(cumulative_reward.cpu().numpy())
            if step % (args.log_interval*10) == 9:
                print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
                logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode mean reward={r_}")
            l_episode_rewards.append(r_)
            
            if args.eval:
                for e in range(num_scenes):
                    episode_done[e].append(True)
                    if len(episode_done[e]) == num_episodes:
                        finished[e] = 1
            l_reward = torch.zeros(num_scenes).to(device)
            cumulative_reward= l_reward
            wait_env = np.zeros((args.num_processes))
            
        # if args.eval:
        #     for e, x in enumerate(done):    # if done, maps from new obs
        #         if x:
        #             episode_done[e].append(True)
        #             if len(episode_done[e]) == num_episodes:
        #                 finished[e] = 1

        #-------------------------------------------------------------------new transition
        ## Sample next action
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
            l_action = np.random.randint(0, 5, num_scenes)
        
        if args.eval:
            
            obs_info = [init_obs_info[e] if lost_goal[e] else obs_info_[e] for e in range(num_scenes)]
            capture = [False]*num_scenes
            for e in range(num_scenes):
                if l_action[e] == 0 or done[e]:
                    capture[e] = True
            envs.save_data(obs_info, capture)
        
        ## transition: next state
        if l_step == args.num_local_steps - 1:
            obs, infos = envs.reset()
            done  = np.array([False]*num_scenes)
            
            images = []
            for n in range(num_scenes):
                pil_img = Image.fromarray(obs[n, :3, ...].cpu().numpy().transpose(1,2,0).astype(np.uint8))
                image = preprocess(pil_img).to(device)
                images.append(image)
            images = torch.stack(images)
            
            goal_idxs = [info['goal_name'] for e, info in enumerate(infos) ]
            init_scores = last_scores = clip_score(images, goal_idxs)
        else:       # if done, obs is next state or current episode
            obs, _, done, infos = envs.step_and_wait(l_action, wait_env)   
        
        if args.eval:
            obs_info_ = envs.get_obs_info()
            if l_step == args.num_local_steps - 1:
                init_obs_info = obs_info_
        
        
        l_action = torch.tensor(l_action)
        
        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if l_step == args.num_local_steps - 1:
            if args.agent == "rl":
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
                l_rollouts.after_update()       # rollout的最后一个state是下一次initial state
            elif args.agent == "random":
                pass
        torch.set_grad_enabled(False)
        
        # ------------------------------------------------------------------
        # Logging:
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

            if len(per_step_l_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_l_rewards),
                        np.median(per_step_l_rewards),
                        np.min(per_step_l_rewards),
                        np.max(per_step_l_rewards))
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
                
            if l_step == args.num_local_steps - 1:
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
        
        if args.agent == "rl":
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
    m = np.array(l_episode_rewards).mean()
    print(f"all episode rewards: {l_episode_rewards}, mean is {m}")
    logging.info(f"all episode rewards: {l_episode_rewards}, mean is {m}")
    
    if args.eval:
        print("Dumping eval details...")
        
if __name__ == "__main__":
    main()