
from src.policy_rl.arguments import get_args
import torch
import numpy as np
import os
import logging
from collections import deque, defaultdict
import gym
import time
from contextlib import contextmanager
from datetime import datetime
from src.policy_rl.envs import make_vec_envs
from src.policy_rl.maps import Maps_Env
from src.policy_rl.utils.storage import GlobalRolloutStorage
from src.policy_rl.model import RL_Policy, RL_Policy2
from src.policy_rl.expert_predictor import ExpertPredictor
from  src.policy_rl import algo 
from src.policy_rl.baseline_frontier import Frontier
from src.policy_rl.trajectory_reward import TrajectoryFeatureReward
import cv2
import json


class StepProfiler:
    def __init__(self, enabled=False, interval=10, use_cuda=False):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def time(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.time()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] += time.time() - start
            self.counts[name] += 1

    def should_report(self, step):
        return self.enabled and step > 0 and step % self.interval == 0

    def report(self, reset=True):
        if not self.totals:
            return ""
        total = sum(self.totals.values())
        rows = []
        for name, seconds in sorted(self.totals.items(), key=lambda item: item[1], reverse=True):
            count = max(1, self.counts[name])
            rows.append(
                "{} {:.3f}s avg {:.4f}s count {} pct {:.1f}%".format(
                    name, seconds, seconds / count, self.counts[name], 100.0 * seconds / max(total, 1e-8)
                )
            )
        if reset:
            self.totals.clear()
            self.counts.clear()
        return "\n\tProfile: " + "\n\t  ".join(rows)

# try:
#     from memory_profiler import profile
# except Exception:
#     # Keep runtime behavior unchanged when memory_profiler is not installed.
#     def profile(func):
#         return func

# @profile
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
    print("Instance detector backend: {}".format(args.detector_backend))
    logging.info("Instance detector backend: %s", args.detector_backend)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")   # 训练的gpu
    profiler = StepProfiler(
        enabled=args.profile_sequence or args.profile_main_sequence,
        interval=args.profile_interval,
        use_cuda=args.cuda,
    )

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
    per_step_traj_rewards = deque(maxlen=1000)
    
    value_losses = deque(maxlen=1000)
    action_losses = deque(maxlen=1000)
    dist_entropies = deque(maxlen=1000)
    distill_losses = deque(maxlen=1000)

    # Starting environments
    torch.set_num_threads(1)
    with profiler.time("env_make"):
        envs = make_vec_envs(args)
    with profiler.time("env_reset"):
        obs_all, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?

    torch.set_grad_enabled(False)

    # Initializing Maps
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    with profiler.time("maps_init"):
        maps = Maps_Env(args)
    trajectory_reward = None
    if args.use_traj_feature_reward:
        trajectory_reward = TrajectoryFeatureReward(
            num_envs=num_scenes,
            map_resolution=args.map_resolution,
            device=device,
            coeff=args.reward_coeff_traj,
            sim_percentile=args.traj_sim_percentile,
            min_region_points=args.traj_min_region_points,
            sim_window=args.traj_sim_window,
            max_points=args.traj_max_points,
            region_update_interval=args.traj_region_update_interval,
            region_method=args.traj_region_method,
            watershed_min_distance=args.traj_watershed_min_distance,
            region_vis_dir=os.path.join(dump_dir, "traj_region_maps") if args.eval else None,
        )
    obs = torch.concat(obs_all, axis=0)
    sensor_pose = [infos[e]['sensor_pose_all'][0] for e in range(num_scenes)]
    orients = [infos[e]['orient_idx'] for e in range(num_scenes)]
    straight = [False for e in range(num_scenes)]
    with profiler.time("initial_map_update"):
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
        with profiler.time("policy_init"):
            policy = RL_Policy2(observation_space.shape, action_space,
                                device = device,
                                use_history=args.use_history_policy,
                                use_action_history_token=args.use_action_history_token,
                                profile_panorama_encoder=(
                                    args.profile_sequence or args.profile_panorama_encoder
                                ),
                                profile_interval=args.profile_interval,
                                ).to(device)

        def _load_checkpoint_into_policy(checkpoint_path):
            print("Loading model {}".format(checkpoint_path))
            logging.info("Loading model {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            if isinstance(checkpoint, dict):
                if 'policy_state_dict' in checkpoint:
                    state_dict = checkpoint['policy_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if args.use_action_history_token:
                incompatible = policy.load_state_dict(state_dict, strict=False)
                unexpected = incompatible.unexpected_keys
                missing_non_history = [
                    key for key in incompatible.missing_keys
                    if not key.startswith("action_history.")
                ]
                if unexpected or missing_non_history:
                    raise RuntimeError(
                        "Checkpoint is incompatible with the base panorama policy: "
                        "missing={}, unexpected={}".format(
                            missing_non_history, unexpected
                        )
                    )
                if incompatible.missing_keys:
                    print(
                        "Initialized new action-history parameters: {}".format(
                            incompatible.missing_keys
                        )
                    )
            else:
                policy.load_state_dict(state_dict)

        if args.load_pretrain != "0":
            _load_checkpoint_into_policy(args.load_pretrain)
        
        agent = algo.PPO(policy, args.clip_param, args.ppo_epoch,
                        args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)
        
        # Initialize Expert Predictor for knowledge distillation
        expert_predictor = None
        if args.use_supervised:
            with profiler.time("expert_init"):
                expert_predictor = ExpertPredictor(device=device, use_semantic_score=args.use_semantic_score)

        
    
        # Storage: 
        if args.use_history_policy and args.use_action_history_token:
            raise ValueError(
                "--use_history_policy and --use_action_history_token "
                "cannot be enabled together"
            )

        rec_state_size = policy.history_token_size
        rollouts = GlobalRolloutStorage(args.num_local_steps,
                                        num_scenes, observation_space.shape,
                                        action_space, rec_state_size,
                                        es,
                                        hidden_size=768,
                                        # hist_len=args.num_local_steps
                                        use_history=args.use_history_policy
                                        ).to(device)
        
        # load weights
        if args.load != "0":
            _load_checkpoint_into_policy(args.load)

        if args.eval:
            policy.eval()

        rec_states = torch.zeros(num_scenes, rec_state_size, device=device)
        extras = torch.zeros(num_scenes, es)
        
        if args.use_history_policy:
            # Get local policy input - 使用 ViT 编码当前全景图
            # 获取 panorama_obs_all 并编码为特征
            with torch.no_grad():
                panorama_obs_list = [infos[i]['panorama_obs_all'] for i in range(num_scenes)]
                # panorama_obs_list[i] 是 dict，包含 'rgb', 'rgb_30', ... 等
                # 转换为 tensor 并编码
                # 获取角度特征
                from src.policy_rl.panorama_model import get_all_point_angle_feature
                ang_feats = get_all_point_angle_feature(policy.network.config.angle_feat_size, )
                
                # 编码每个环境的全景图
                curr_pano_img_feats_list = []
                curr_pano_ang_feats_list = []
                for e in range(num_scenes):
                    # 获取当前环境的全景图
                    obs_dict = panorama_obs_list[e]
                    # 提取 rgb 图像 (12 views)
                    rgb_types = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 
                                'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210',
                                'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
                    images = []
                    for rgb_type in rgb_types:
                        img = obs_dict[rgb_type]
                        images.append(img)
                    # 转换为 tensor (12, H, W, 3)
                    images = np.stack(images, axis=0)
                    images = torch.from_numpy(images).float() / 255.0
                    # 编码
                    img_feats = policy.network.encoding(images.unsqueeze(0).to(device))
                    curr_pano_img_feats_list.append(img_feats.squeeze(0))  # (12, 768)
                    # 角度特征
                    curr_pano_ang_feats_list.append(ang_feats.to(device))  # (12, 2)
                
                # 合并所有环境的特征
                curr_pano_img_feats = torch.stack(curr_pano_img_feats_list, dim=0)  # (num_scenes, 12, 768)
                curr_pano_ang_feats = torch.stack(curr_pano_ang_feats_list, dim=0)  # (num_scenes, 12, 2)
                
                # 将当前观测的特征存储到 rollouts (step 0)
                rollouts.pano_img_feats[0] = curr_pano_img_feats
                rollouts.pano_ang_feats[0] = curr_pano_ang_feats
        else:
            curr_pano_img_feats = None
            curr_pano_ang_feats = None
            local_input = [torch.from_numpy(info['panorama_obs']).unsqueeze(0) for info in infos]
            local_input = torch.concat(local_input, axis=0)
            rollouts.obs[0].copy_(local_input)   # 
            rollouts.extras[0].copy_(extras)
            del local_input
        
        with profiler.time("initial_policy_act"):
            if args.use_action_history_token:
                value, action, action_log_prob, rec_states = \
                    policy.act_with_history_token(
                        rollouts.obs[0],
                        rec_states,
                        rollouts.masks[0],
                        deterministic=False,
                    )
            else:
                value, action, action_log_prob = policy.act(
                    rollouts.obs[0] if not args.use_history_policy else None,  # 如果使用历史模型，初始时不使用当前观测作为输入
                    extras=extras,
                    deterministic=False,
                    curr_pano_img_feats=curr_pano_img_feats,
                    curr_pano_ang_feats=curr_pano_ang_feats,
                    hist_pano_img_feats=None,
                    hist_pano_ang_feats=None,
                    hist_actions=None,
                    hist_masks=None,
                    compute_hist_embed=False
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
                
    ## Env transition:
    # pred instance, get semantic masks and step env: 
    # Get expert_probs from ExpertPredictor using panorama_obs_all (batch inference)
    expert_probs_batch = None
    if args.use_supervised:
        panorama_obs_list = [infos[i]['panorama_obs_all'] for i in range(num_scenes)]
        
        with profiler.time("initial_supervised_predict"):
            expert_probs_batch = expert_predictor.predict(panorama_obs_list)
            
    # print(f"action is {l_action}")
    with profiler.time("initial_env_step"):
        obs_all, _, done, infos = envs.step_and_preprocess(action, vis_inputs_frames)
    action = torch.tensor(action)
        
    with profiler.time("initial_post_step_map_update"):
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
    traj_feat_reward = torch.zeros(num_scenes).to(device)

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
                
        with profiler.time("reward_base"):
            # get reward: map change after state transition
            if done[0]:     # maps are new obs, sum of map will be small, and get negative reward
                l_reward = last_reward
            else:
                sequence_r =  args.reward_coeff_seq * torch.tensor([infos[e]['sequence_reward'] for e in range(num_scenes)]).to(last_reward.device)
                cls_entropy_r = args.reward_coeff_ce *  torch.tensor([infos[e]['cls_etp'] for e in range(num_scenes)]).to(last_reward.device)
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
        with profiler.time("local_input_prep"):
            locs = full_pose.cpu().numpy()
            
            if args.agent == "rl":
                # for e in range(num_scenes):
                #     local_orientation[e] = int((locs[e, 2] + 180.0) / 5.)   # 
                #     local_xy[e] = torch.from_numpy(locs[e, :2][np.newaxis, :])
                
                if args.use_history_policy:
                    # 获取当前观测的全景特征
                    with torch.no_grad():
                        panorama_obs_list = [infos[i]['panorama_obs_all'] for i in range(num_scenes)]
                        from src.policy_rl.panorama_model import get_all_point_angle_feature
                        ang_feats = get_all_point_angle_feature(policy.network.config.angle_feat_size, )
                        
                        curr_pano_img_feats_list = []
                        curr_pano_ang_feats_list = []
                        for e in range(num_scenes):
                            obs_dict = panorama_obs_list[e]
                            rgb_types = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 
                                        'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210',
                                        'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
                            images = []
                            for rgb_type in rgb_types:
                                img = obs_dict[rgb_type]
                                images.append(img)
                            images = np.stack(images, axis=0)
                            images = torch.from_numpy(images).float() / 255.0
                            img_feats = policy.network.encoding(images.unsqueeze(0).to(device))
                            curr_pano_img_feats_list.append(img_feats.squeeze(0))
                            curr_pano_ang_feats_list.append(ang_feats.to(device))
                        
                        curr_pano_img_feats = torch.stack(curr_pano_img_feats_list, dim=0)
                        curr_pano_ang_feats = torch.stack(curr_pano_ang_feats_list, dim=0)
                    
                    local_input = [torch.from_numpy(info['panorama_obs']).unsqueeze(0) for info in infos]
                    local_input = torch.concat(local_input, axis=0)
                else:
                    curr_pano_img_feats = None
                    curr_pano_ang_feats = None
                    local_input = [torch.from_numpy(info['panorama_obs']).unsqueeze(0) for info in infos]
                    local_input = torch.concat(local_input, axis=0)
            # extras[:, 0] = local_orientation[:, 0]
            # extras[:, :2] = local_xy[:]
            # print(f"input sxtras: {extras}")
        # Add samples to local policy storage
        reward = l_reward - last_reward + cls_entropy_r + sequence_r
        # reward = l_reward - last_reward
        traj_reward = torch.zeros(num_scenes, device=device)
        if (
            args.agent == "rl"
            and trajectory_reward is not None
            and hasattr(policy.network, "ob_img_feats")
            and policy.network.ob_img_feats is not None
        ):
            with profiler.time("traj_reward"):
                traj_reward = trajectory_reward.update_and_compute(
                    maps=maps,
                    actions=action,
                    img_feats=policy.network.ob_img_feats,
                    done=done,
                )
            reward = reward + traj_reward
            step_other_reward += traj_reward
            traj_feat_reward += traj_reward
            if args.log_traj_reward_detail:
                dbg = trajectory_reward.get_last_debug(args.log_traj_reward_env)
                if dbg is not None:
                    traj_log = (
                        "[traj_reward_debug] step={step} local_step={local_step} env={env} "
                        "traj_step={traj_step} reason={reason} action={action} region={region} "
                        "cell={cell} candidates={candidates} hist_sims={hist_sims} "
                        "max_sim={max_sim} threshold={threshold} sim_min={sim_min} sim_max={sim_max} "
                        "raw_reward={raw_reward:.6f} scaled_reward={scaled_reward:.6f}"
                    ).format(
                        step=step,
                        local_step=l_step,
                        env=int(args.log_traj_reward_env),
                        traj_step=dbg.get("step", None),
                        reason=dbg.get("reason", "none"),
                        action=dbg.get("action", None),
                        region=dbg.get("region", None),
                        cell=dbg.get("cell", None),
                        candidates=dbg.get("num_candidates", None),
                        hist_sims=dbg.get("num_hist_sims", None),
                        max_sim=dbg.get("max_sim", None),
                        threshold=dbg.get("threshold", None),
                        sim_min=dbg.get("sim_min", None),
                        sim_max=dbg.get("sim_max", None),
                        raw_reward=float(dbg.get("raw_reward", 0.0)),
                        scaled_reward=float(dbg.get("scaled_reward", 0.0)),
                    )
                    print(traj_log)
                    logging.info(traj_log)
        
        if args.agent == "rl":
            
            with profiler.time("rollout_insert"):
                rollouts.insert(
                        local_input, rec_states,      # state_t+1
                        action, action_log_prob, value,   # action, reward_t
                        reward, l_masks, extras,
                        expert_probs=expert_probs_batch,
                        pano_img_feats=curr_pano_img_feats,
                        pano_ang_feats=curr_pano_ang_feats
                    )
            del local_input
        last_reward = l_reward

        # 
        reward_incre_mean = np.mean(reward.cpu().numpy())
        reward_curr_mean = np.mean(l_reward.cpu().numpy())
        per_step_incre_rewards.append(reward_incre_mean)
        per_step_curr_rewards.append(reward_curr_mean)
        if args.use_traj_feature_reward:
            per_step_traj_rewards.append(np.mean(traj_reward.cpu().numpy()))

        # print(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        # logging.info(f"step-{step} local-{l_step} reward:{l_reward_mean}, sum reward:{reward_mean}")
        
        if done[0]:
            r_ = np.mean(l_reward.cpu().numpy())
            r_so = np.mean(step_other_reward.cpu().numpy())
            r_tf = np.mean(traj_feat_reward.cpu().numpy())
            print(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode map reward={r_}\nepisode not map cumu reward={r_so}\nepisode all reward={r_+r_so}\nepisode traj reward={r_tf}")
            logging.info(f"episode over in {step} step, {l_step} local step, rollouts done;\n episode map reward={r_}\nepisode not map cumu reward={r_so}\nepisode all reward={r_+r_so}\nepisode traj reward={r_tf}")
            episode_rewards.append(r_)

            l_reward = torch.zeros(num_scenes).to(device)
            step_other_reward = torch.zeros(num_scenes).to(device)
            last_reward = l_reward
            traj_feat_reward = torch.zeros(num_scenes).to(device)
            
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
            with profiler.time("policy_act"):
                if args.use_history_policy and rollouts.step > 0:
                    # 调用 act() 时传入历史特征
                    # 获取当前步之前的所有全景特征作为历史特征
                    hist_pano_img_feats, hist_pano_ang_feats = rollouts.get_all_pano_feats()
                    hist_pano_img_feats = hist_pano_img_feats.detach().transpose(1,0)  # (num_scenes, his_len, views, image_feat_size)
                    hist_pano_ang_feats = hist_pano_ang_feats.detach().transpose(1,0) 
                
                    # 获取历史动作
                    hist_actions = rollouts.actions[:rollouts.step].transpose(1,0).squeeze(-1)  # (num_scenes, his_len)
                    
                    # 历史掩码（所有历史位置都是有效的）
                    hist_masks = torch.ones(num_scenes, rollouts.step, dtype=torch.bool, device=device)
                else:
                    hist_pano_img_feats = None
                    hist_pano_ang_feats = None
                    hist_actions = None
                    hist_masks = None
                # 这样会返回当前观测的特征，用于下一步
                if args.use_action_history_token:
                    value, action, action_log_prob, rec_states = \
                        policy.act_with_history_token(
                            rollouts.obs[l_step + 1],
                            rec_states,
                            l_masks,
                            deterministic=False,
                        )
                else:
                    value, action, action_log_prob = policy.act(
                        rollouts.obs[l_step + 1] if not args.use_history_policy else None, 
                        extras=None,
                        deterministic=False,
                        curr_pano_img_feats=curr_pano_img_feats,
                        curr_pano_ang_feats=curr_pano_ang_feats,
                        hist_pano_img_feats=hist_pano_img_feats,
                        hist_pano_ang_feats=hist_pano_ang_feats,
                        hist_actions=hist_actions,
                        hist_masks=hist_masks,
                        compute_hist_embed=False
                        )
            
            action = action.cpu().numpy()
            
        elif args.agent == "random":
            action = np.random.randint(0, 12, num_scenes)        
        if args.agent == "frontier":  # must be after updating vis_inputs
            action, goals, short_time_goals = policy.get_actions(vis_inputs)        
            if args.visualize or args.print_images:
                for e, p_input in enumerate(vis_inputs):
                    p_input["frontier_goal"] = goals[e]
                    p_input["short_time_goal"] = short_time_goals[e]
        
        
        # transition: next state
        # pred instance, get semantic masks and step env
        # Get expert_probs from ExpertPredictor using panorama_obs_all (batch inference)
        expert_probs_batch = None
        if args.use_supervised:
            panorama_obs_list = [infos[i]['panorama_obs_all'] for i in range(num_scenes)]
            
            # Test: set the 3rd rgb (rgb_60, index 2 in angles list) of the first environment to all zeros
            # panorama_obs_list[0]['rgb_60'] = np.zeros_like(panorama_obs_list[0]['rgb_60'])
            # panorama_obs_list[0]['depth_60'] = np.zeros_like(panorama_obs_list[0]['depth_60'])
            
            with profiler.time("supervised_predict"):
                expert_probs_batch = expert_predictor.predict(panorama_obs_list)
            
        # print(f"action is {l_action}")
        with profiler.time("env_step"):
            obs_all, _, done, infos = envs.step_and_preprocess(action, vis_inputs_frames)    # if done ,envs.reset, obs are ones after reset
        action = torch.tensor(action)
        # if episode over, reset maps
        for e, x in enumerate(done):    # if done, maps from new obs
            if x:
                maps._init_map_and_pose_for_env(e)
                print(f"Env {e}'s episode over in {step} step, {l_step} local step, reset maps")
        
        
        with profiler.time("map_update"):
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
        with profiler.time("ppo_update"):
            if l_step == args.num_local_steps - 1:
            # if l_step == 100 - 1:
                if not args.eval and args.agent == "rl":
                    if args.use_action_history_token:
                        # obs[-1] and rec_states[-1] describe the same state:
                        # the final observation and its history before sampling
                        # an action from that observation.
                        next_value = policy.get_value(
                            rollouts.obs[-1],
                            history_token=rollouts.rec_states[-1],
                            masks=rollouts.masks[-1],
                        ).detach()
                    else:
                        next_value = policy.get_value(
                            None if args.use_history_policy else rollouts.obs[-1],
                            extras=None,
                            curr_pano_img_feats=rollouts.pano_img_feats[-1] if args.use_history_policy else None,
                            curr_pano_ang_feats=rollouts.pano_ang_feats[-1] if args.use_history_policy else None,
                        ).detach()
                    rollouts.compute_returns(next_value, args.use_gae,
                                               args.gamma, args.tau)
                    
                    if args.use_supervised:
                        value_loss, action_loss, dist_entropy, distill_loss = \
                            agent.update_with_supervise(rollouts)
                        distill_losses.append(distill_loss)
                    else:
                        value_loss, action_loss, dist_entropy = \
                            agent.update(rollouts)
                    value_losses.append(value_loss)
                    action_losses.append(action_loss)
                    dist_entropies.append(dist_entropy)
                    
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
                    " per step mean/med/min/max, semantic map rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_curr_rewards),
                        np.median(per_step_curr_rewards),
                        np.min(per_step_curr_rewards),
                        np.max(per_step_curr_rewards))
                ])
            if len(per_step_incre_rewards) > 0:
                log += " ".join([
                    " per step mean/med/min/max, step rew:",
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f},".format(
                        np.mean(per_step_incre_rewards),
                        np.median(per_step_incre_rewards),
                        np.min(per_step_incre_rewards),
                        np.max(per_step_incre_rewards))
                ])
            if args.use_traj_feature_reward and len(per_step_traj_rewards) > 0:
                log += " traj_penalty_mean: {:.4f},".format(
                    np.mean(per_step_traj_rewards)
                )
                sim_stats = trajectory_reward.get_similarity_stats() if trajectory_reward is not None else None
                if sim_stats is not None:
                    log += " traj_sim_min/max: {:.4f}/{:.4f},".format(
                        sim_stats[0], sim_stats[1]
                    )

            log += "\n\tLosses:"
            if len(value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(value_losses),
                        np.mean(action_losses),
                        np.mean(dist_entropies))
                ])
                if args.use_supervised:
                    log += " Distill Loss: {:.3f},".format(np.mean(distill_losses))
            
            if profiler.should_report(step):
                log += profiler.report(reset=True)
                
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
    
    if args.eval:
        print("Dumping eval details...")
        

        
if __name__ == "__main__":
    main()
