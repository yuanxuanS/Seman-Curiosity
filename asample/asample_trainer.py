from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry

import gc
import os
import sys
import random
import warnings
from collections import defaultdict, deque
from typing import Dict, List
import jsonlines
from gym import spaces
import contextlib

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from asample.common.env_utils import construct_envs, is_slurm_batch_job
from .utils import get_camera_orientations12

from habitat.utils import profiling_wrapper
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    rank0_only,
    load_resume_state,
    save_resume_state,
    requeue_job
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from asample.models.rollout_storage import GraphRolloutStorage
from asample.common.ops import pad_tensors_wgrad, gen_seq_masks
from asample.models.graph_utils import GraphMap
import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence


@baseline_registry.register_trainer(name="asample")
class ActiveSampleTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.RL.episode_length) #  
        self.local_rank = self.config.local_rank
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        # self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        # self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        # self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH",]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space
    
    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True
        if is_slurm_batch_job():
            add_signal_handlers()
        
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        self.ppo_cfg = self.config.RL.PPO
        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from asample.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data_scene/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data_scene/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        # self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)
        
        
        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.RL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            # if config.IL.is_requeue:
            #     self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.policy,
            clip_param=self.ppo_cfg.clip_param,
            ppo_epoch=self.ppo_cfg.ppo_epoch,
            num_mini_batch=self.ppo_cfg.num_mini_batch,
            value_loss_coef=self.ppo_cfg.value_loss_coef,
            entropy_coef=self.ppo_cfg.entropy_coef,
            lr=self.ppo_cfg.lr,
            eps=self.ppo_cfg.eps,
            max_grad_norm=self.ppo_cfg.max_grad_norm,
            use_normalized_advantage=self.ppo_cfg.use_normalized_advantage,
        )
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)
        
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
         
        self.lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )
        return start_iter
    
    def train(self):
        self._set_config()

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.RL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        
        self.ppo_cfg = self.config.RL.PPO
        self.rollouts = GraphRolloutStorage(
            # ppo_cfg.num_steps,
            self.max_len,
            self.envs.num_envs,
            self.envs.action_spaces[0],
            # ppo_cfg.hidden_size,
            # num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=self.ppo_cfg.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)
        
        total_iter = self.config.RL.iters
        log_every  = self.config.RL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.pth_time = 0.0
        self.prev_time = 0
        self.env_time = 0.0
        self.t_start = time.time()
        
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.ppo_cfg.reward_window_size)
        )
        # self.scaler = GradScaler()      # 加速训练
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            logs = self._train_interval(interval,)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)
    
    def _train_interval(self, interval,):
        # self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            for idx in pbar:
                # self.optimizer.zero_grad()
                # self.loss = 0.

                losses = self.rollout('train',)
                # 更新的改在这里？ TODO
                # self.scaler.scale(self.loss).backward() # self.loss.backward()
                # # self.scaler.step(self.optimizer)        # self.optimizer.step()
                # self.scaler.update()
                self._training_log(writer, losses)
                
                if self.local_rank < 1:
                    pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)


    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], 
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + self.prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
            
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)
            
            gmap_vp_ids = ghost_vp_ids
            # gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            # gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            # gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            # gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
            #                [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            # gmap_img_fts = torch.stack(
            #     [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            # )
            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            # gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            # for j in range(1, len(gmap_vp_ids)):
            #     for k in range(j+1, len(gmap_vp_ids)):
            #         vp1 = gmap_vp_ids[j]
            #         vp2 = gmap_vp_ids[k]
            #         if not vp1.startswith('g') and not vp2.startswith('g'):
            #             dist = gmap.shortest_dist[vp1][vp2]
            #         elif not vp1.startswith('g') and vp2.startswith('g'):
            #             front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
            #             dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
            #         elif vp1.startswith('g') and vp2.startswith('g'):
            #             front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
            #             front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
            #             dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
            #         else:
            #             raise NotImplementedError
            #         gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            # batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            # batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists)) # 距离矩阵
            # batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        # batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        # batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        # gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        # for i in range(bs):
        #     gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        # gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 
            # 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 
            'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 
            # 'gmap_visited_masks': batch_gmap_visited_masks, 
            # 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def pre_obs(self, batch, mode, prev_vp, stepk):
        # cand waypoint prediction
        wp_outputs = self.policy.net(
            mode = "waypoint",
            waypoint_predictor = self.waypoint_predictor,
            observations = batch,
            in_train = (mode == 'train'),
        )
        # pano encoder
        vp_inputs = self._vp_feature_variable(wp_outputs)
        vp_inputs.update({
            'mode': 'panorama',
        })
        pano_embeds, pano_masks = self.policy.net(**vp_inputs)
        avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                            torch.sum(pano_masks, 1, keepdim=True)

        # get vp_id, vp_pos of cur_node and cand_ndoe
        cur_pos, cur_ori = self.get_pos_ori()
        cur_vp, cand_vp, cand_pos = [], [], []
        for i in range(self.envs.num_envs):
            cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(      # 根据waypoint predictor预测的点，得到候选点的pos
                cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
            )
            cur_vp.append(cur_vp_i)
            cand_vp.append(cand_vp_i)
            cand_pos.append(cand_pos_i)
        
        if mode == 'train' or self.config.VIDEO_OPTION:
            cand_real_pos = []
            for i in range(self.envs.num_envs):
                cand_real_pos_i = [
                    self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                    for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                ]
                cand_real_pos.append(cand_real_pos_i)
        else:
            cand_real_pos = [None] * self.envs.num_envs
        
        for i in range(self.envs.num_envs):
            cur_embeds = avg_pano_embeds[i]
            cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]      # nav_types 0/1 是否是有效cand
            self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                        cur_vp[i], cur_pos[i], cur_embeds,
                                        cand_vp[i], cand_pos[i], cand_embeds,
                                        cand_real_pos[i])
        nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
        nav_inputs.update({
            'mode': 'navigation',
            # 'txt_embeds': txt_embeds,
            # 'txt_masks': txt_masks,
        })
        no_vp_left = nav_inputs.pop('no_vp_left')
        
        return nav_inputs, no_vp_left,cur_vp
    
    def _build_actions(self, cpu_a_t, stepk, no_vp_left, cur_vp, nav_inputs, prev_vp):
        # make equiv action
        env_actions = []
        use_tryout = (self.config.RL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
        for i, gmap in enumerate(self.gmaps):
            # if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:   # stop
            # if stepk == self.max_len - 1 or no_vp_left[i]:   # stop
            #     # stop at node with max stop_prob
            #     vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
            #     stop_scores = [s[1] for s in vp_stop_scores]
            #     stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
            #     stop_pos = gmap.node_pos[stop_vp]
            #     if self.config.RL.back_algo == 'control':
            #         back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
            #         back_path = back_path[1:]
            #     else:
            #         back_path = None
            #     vis_info = {
            #             'nodes': list(gmap.node_pos.values()),
            #             'ghosts': list(gmap.ghost_aug_pos.values()),
            #             'predict_ghost': stop_pos,
            #     }
            #     env_actions.append(
            #         {
            #             'action': {
            #                 'act': 0,
            #                 'cur_vp': cur_vp[i],
            #                 'stop_vp': stop_vp, 'stop_pos': stop_pos,
            #                 'back_path': back_path,
            #                 'tryout': use_tryout,
            #             },
            #             'vis_info': vis_info,
            #         }
            #     )
            # else:
            ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
            ghost_pos = gmap.ghost_aug_pos[ghost_vp]
            _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
            front_pos = gmap.node_pos[front_vp]
            if self.config.VIDEO_OPTION:
                # teacher_action_cpu = teacher_actions[i].cpu().item()
                # if teacher_action_cpu in [0, -100]:
                #     teacher_ghost = None
                # else:
                #     teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                vis_info = {
                    'nodes': list(gmap.node_pos.values()),
                    'ghosts': list(gmap.ghost_aug_pos.values()),
                    'predict_ghost': ghost_pos,
                    # 'teacher_ghost': teacher_ghost,
                }
            else:
                vis_info = None
            # teleport to front, then forward to ghost; teleport是模拟场景直接将agent传送到goal相邻的mesh，然后
            if self.config.RL.back_algo == 'control':   # 控制模态：走图上的最短路径
                back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                back_path = back_path[1:]
            else:       # 传送模态
                back_path = None
            env_actions.append(
                {
                    'action': {
                        'act': 4,
                        'cur_vp': cur_vp[i],
                        'front_vp': front_vp, 'front_pos': front_pos,
                        'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                        'back_path': back_path,
                        'tryout': use_tryout,
                    },
                    'vis_info': vis_info,
                }
            )
            prev_vp[i] = front_vp
            if self.config.MODEL.consume_ghost:     # 原文提到，为了避免重复选到不了的节点，选择ghost后先删除，再走去
                gmap.delete_ghost(ghost_vp)
        return env_actions, prev_vp
        
        
    def _update_agent(self,):
        buffer_index = 0
        t_update_model = time.time()
        with torch.no_grad():
            env_slice = slice(
                    int(buffer_index * self.envs.num_envs / self.rollouts._nbuffers),
                    int((buffer_index + 1) * self.envs.num_envs / self.rollouts._nbuffers),
                )
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]
            gmap_vp_ids = self.rollouts.gmap_vp_ids[ \
                    self.rollouts.current_rollout_step_idxs[buffer_index]][env_slice]
                
            gmap_masks_ = step_batch['gmap_masks']
            gmap_img_fts_ = step_batch['gmap_img_fts']
            gmap_pos_fts_ = step_batch['gmap_pos_fts']
            valid_ghost_size = int(gmap_masks_.sum(1).max().cpu())
            gmap_masks = gmap_masks_[:, :valid_ghost_size]
            gmap_img_fts = gmap_img_fts_[:, :valid_ghost_size]
            gmap_pos_fts = gmap_pos_fts_[:, :valid_ghost_size]
            obs = {
                'gmap_img_fts': gmap_img_fts,
                'gmap_pos_fts': gmap_pos_fts,
                'gmap_masks': gmap_masks,
                'gmap_vp_ids': gmap_vp_ids
                }
            next_value = self.policy.get_value(**obs)

        self.rollouts.compute_returns(
            next_value, self.ppo_cfg.use_gae, self.ppo_cfg.gamma, self.ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)
    
    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )
        
    def rollout(self, mode):
        self.current_episode_reward = torch.zeros_like(self.current_episode_reward)
        self.env_time = 0.0
        self.pth_time = 0.0
        self.num_steps_done = 0     # 每个episode的node step数（环境累积）
        
        self.envs.resume_all()
        observations = self.envs.reset()
        # instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        # instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        # observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        #                                           max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)


        loss = 0.
        total_actions = 0.
        not_done_masks = list(range(self.envs.num_envs))
        buffer_index = 0
        max_ghost_node_size = 200
        
        count_steps_delta = 0       # 统计 node step数（每个环境累加step）
        
        
        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.RL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.RL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        dones_ = [False] * self.envs.num_envs
        
        self.policy.eval()
        
        t_sample_action = time.time()
        with torch.no_grad():
            env_slice = slice(
                    int(buffer_index * self.envs.num_envs / self.rollouts._nbuffers),
                    int((buffer_index + 1) * self.envs.num_envs / self.rollouts._nbuffers),
                )
            ## init state
            nav_inputs, no_vp_left, cur_vp = self.pre_obs(batch, mode, prev_vp, stepk=0)
            
            gmap_img_fts_size = nav_inputs['gmap_img_fts'].shape[1]
            gmap_img_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size , 768), dtype=torch.float32).to(self.device)
            gmap_pos_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size , 7), dtype=torch.float32).to(self.device)
            gmap_masks_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size), dtype=torch.bool).to(self.device)
            
            gmap_img_fts = torch.concat((nav_inputs['gmap_img_fts'], gmap_img_pad), dim=1)
            gmap_pos_fts = torch.concat((nav_inputs['gmap_pos_fts'], gmap_pos_pad), dim=1)
            gmap_masks = torch.concat((nav_inputs['gmap_masks'], gmap_masks_pad), dim=1)
            obs = {
                'gmap_img_fts': gmap_img_fts,
                'gmap_pos_fts': gmap_pos_fts,
                'gmap_masks': gmap_masks,
                'gmap_vp_ids': nav_inputs['gmap_vp_ids']
                }
            self.rollouts.buffers["gmap_img_fts"][0] = obs['gmap_img_fts'].detach()
            self.rollouts.buffers["gmap_pos_fts"][0] = obs['gmap_pos_fts']
            self.rollouts.buffers["gmap_masks"][0] = obs['gmap_masks']
            self.rollouts.gmap_vp_ids[0] = nav_inputs['gmap_vp_ids']
            
            values, actions, actions_log_probs = self.policy.act(**obs)
            

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs

            cpu_a_t = actions.cpu().numpy()

            env_actions, prev_vp = self._build_actions(cpu_a_t, stepk, no_vp_left,
                                              cur_vp, nav_inputs, prev_vp)
            self.pth_time += time.time() - t_sample_action
            
            t_step_env = time.time()
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            self.env_time += time.time() - t_step_env
            
            # done mask
            dones_ = [dones[i] or no_vp_left[i] or dones_[i] for i in range(len(dones))]
            done_masks = torch.tensor(
                [[d] for d in dones_],
                dtype=torch.bool,
            )
            not_done_masks = torch.logical_not(done_masks)
            
            # rewards:
            rewards = - torch.tensor([info['path_length'] for info in infos], dtype=torch.float32).unsqueeze(1) * 0.01
            
            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # 当 done=True时，才+
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # 记录episode个数

            self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    # gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    # metric['steps_taken'] = info['steps_taken']
                    # metric['distance_to_goal'] = distances[-1]
                    # metric['success'] = 1. if distances[-1] <= 3. else 0.
                    # metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    # metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    # metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    # dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    # metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    # metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env TODO
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_masks = torch.concat((not_done_masks[:i], not_done_masks[i+1:]))
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break
            
            t_update_stats = time.time()
            # obs for next step
            # observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            nav_inputs, no_vp_left, cur_vp = self.pre_obs(batch, mode, prev_vp, stepk=stepk)
                    
            gmap_img_fts_size = nav_inputs['gmap_img_fts'].shape[1]
            gmap_img_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size , 768), dtype=torch.float32).to(self.device)            
            gmap_pos_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size , 7), dtype=torch.float32).to(self.device)
            gmap_masks_pad = torch.zeros((self.envs.num_envs, max_ghost_node_size - gmap_img_fts_size), dtype=torch.bool).to(self.device)
            
            gmap_img_fts = torch.concat((nav_inputs['gmap_img_fts'], gmap_img_pad), dim=1)
            gmap_pos_fts = torch.concat((nav_inputs['gmap_pos_fts'], gmap_pos_pad), dim=1)
            gmap_masks = torch.concat((nav_inputs['gmap_masks'], gmap_masks_pad), dim=1)

            obs = {
                'gmap_img_fts': gmap_img_fts.detach(),
                'gmap_pos_fts': gmap_pos_fts,
                'gmap_masks': gmap_masks,
                'gmap_vp_ids': nav_inputs['gmap_vp_ids'],
                }
            self.rollouts.insert(
                next_nav_inputs=obs,
                actions=actions.unsqueeze(-1),
                action_log_probs=actions_log_probs,
                value_preds=values,
                rewards=rewards.to(self.device),
                next_masks=torch.tensor(not_done_masks).to(self.device),
                
            )
            self.rollouts.advance_rollout()
            
            self.pth_time += time.time() - t_update_stats

            t_sample_action = time.time()
            with torch.no_grad():
                env_slice = slice(
                    int(buffer_index * self.envs.num_envs / self.rollouts._nbuffers),
                    int((buffer_index + 1) * self.envs.num_envs / self.rollouts._nbuffers),
                )
                count_steps_delta += env_slice.stop - env_slice.start
                step_batch = self.rollouts.buffers[
                    self.rollouts.current_rollout_step_idxs[buffer_index],
                    env_slice,
                ]
                gmap_vp_ids = self.rollouts.gmap_vp_ids[ \
                    self.rollouts.current_rollout_step_idxs[buffer_index]][env_slice]
                
                gmap_masks_ = step_batch['gmap_masks']
                gmap_img_fts_ = step_batch['gmap_img_fts']
                gmap_pos_fts_ = step_batch['gmap_pos_fts']
                valid_ghost_size = int(gmap_masks_.sum(1).max().cpu())
                gmap_masks = gmap_masks_[:, :valid_ghost_size]
                gmap_img_fts = gmap_img_fts_[:, :valid_ghost_size]
                gmap_pos_fts = gmap_pos_fts_[:, :valid_ghost_size]
                obs = {
                    'gmap_img_fts': gmap_img_fts,
                    'gmap_pos_fts': gmap_pos_fts,
                    'gmap_masks': gmap_masks,
                    'gmap_vp_ids': gmap_vp_ids
                    }
                values, actions, actions_log_probs = self.policy.act(**obs)


            

        if mode == 'train':
            # loss = ml_weight * loss / total_actions
            # self.loss += loss
            # self.logs['IL_loss'].append(loss.item())
            (
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent()
            
            if self.ppo_cfg.use_linear_lr_decay:
                self.lr_scheduler.step()  # type: ignore

            self.num_updates_done += 1
            losses = self._coalesce_post_step(
                dict(value_loss=value_loss, action_loss=action_loss, dist_entropy=dist_entropy),
                count_steps_delta,      # 每个环境的每步为1 step
            )
        return losses
            
