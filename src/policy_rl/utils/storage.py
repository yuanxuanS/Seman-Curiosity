# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        if not self.use_history:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0
        self.has_extras = False
        self.extras_size = None

        self.expert_probs_size = 12
    def reset(self):
        if not self.use_history:
            self.obs = torch.zeros_like(self.obs)
        self.rec_states = torch.zeros_like(self.rec_states)
        self.rewards = torch.zeros_like(self.rewards)
        self.value_preds = torch.zeros_like(self.value_preds)
        self.returns = torch.zeros_like(self.returns)
        self.action_log_probs = torch.zeros_like(self.action_log_probs)
        self.actions = torch.zeros_like(self.actions, dtype=self.actions.dtype)
        self.masks = torch.ones_like(self.masks)
        self.step = 0
        self.has_extras = False
        self.extras_size = None
        if hasattr(self, 'expert_probs'):
            self.expert_probs = torch.zeros_like(self.expert_probs)
        
    def to(self, device):
        if not self.use_history:
            self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        if hasattr(self, 'expert_probs'):
            self.expert_probs = self.expert_probs.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        if not self.use_history:
            self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        if not self.use_history:
            self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            if self.use_history:
                T_max = self.pano_img_feats.size(0) - 1 
                num_envs = self.pano_img_feats.size(1)
                device = self.pano_img_feats.device
                
                indices_mask = torch.tensor(indices, device=device)
                env_indices = indices_mask % num_envs   # 形状: [batch_size]
                step_indices = indices_mask // num_envs # 形状: [batch_size]，代表当前是第几步
                # 1. 创建一个时间轴 [0, 1, 2, ..., T_max - 1]
                time_steps = torch.arange(T_max, device=device).view(1, -1) # [1, T_max]
                # 2. 将每个样本的当前步长扩展为 [batch_size, 1]
                current_steps = step_indices.view(-1, 1) # [batch_size, 1]
                # 3. 广播比较：如果时间轴上的点 <= 当前步，则为 True
                # 这确保了对于第 i 个样本，只有 0 到 step_indices[i] 的位置是 True
                batch_hist_masks = (time_steps <= current_steps)        # env*steplen, steplen
                
                # 1. 使用 env_indices 提取对应的环境序列
                # 提取后的形状: [T_max, batch_size, 12, 768]
                hist_img_all_envs = self.pano_img_feats[:-1][:, env_indices]
                hist_ang_all_envs = self.pano_ang_feats[:-1][:, env_indices]
                # 2. 转置维度，使 batch_size 成为第一个维度
                # 目标形状: [batch_size, T_max, 12, 768]
                hist_img_final = hist_img_all_envs.permute(1, 0, 2, 3)
                hist_ang_final = hist_ang_all_envs.permute(1, 0, 2, 3)
                # 3. 对 actions 进行同样处理 (假设 actions 原始为 [T_max, num_envs, n_actions])
                # 提取后转置为 [batch_size, T_max, n_actions]
                hist_actions_final = self.actions[:, env_indices].permute(1, 0, 2)
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices] if not self.use_history else None,
                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                    if self.has_extras else None,
                'expert_probs': self.expert_probs.view(
                    -1, self.expert_probs_size)[indices]
                    if hasattr(self, 'expert_probs') else None,
                'curr_pano_img_feats': self.pano_img_feats[:-1].view(-1,
                *self.pano_img_feats.size()[2:])[indices] if hasattr(self, 'pano_img_feats') else None,
                'curr_pano_ang_feats': self.pano_ang_feats[:-1].view(-1,
                *self.pano_ang_feats.size()[2:])[indices] if hasattr(self, 'pano_ang_feats') else None,
                'hist_pano_img_feats': hist_img_final if self.use_history else None,
                'hist_pano_ang_feats': hist_ang_final if self.use_history else None,
                'hist_actions': hist_actions_final if self.use_history else None,   
                'hist_masks':  batch_hist_masks if self.use_history else None,            
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []
            if hasattr(self, 'expert_probs'):
                expert_probs = []
    
            for offset in range(num_envs_per_batch):
                if start_ind + offset > num_processes - 1:
                    break
                ind = perm[start_ind + offset]
                # obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])
                if hasattr(self, 'expert_probs'):
                    expert_probs.append(self.expert_probs[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)
            if hasattr(self, 'expert_probs'):
                expert_probs = torch.stack(expert_probs, 1)
            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
                'expert_probs': _flatten_helper(T, N, expert_probs) if hasattr(self, 'expert_probs') else None,
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(
        self, 
        num_steps, 
        num_processes, 
        obs_shape, 
        action_space,
        rec_state_size, 
        extras_size,
        expert_probs_size=12,
        hidden_size=768,
        use_history=False,
    ):
        self.use_history = use_history
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space, rec_state_size)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.has_extras = False
        self.extras_size = extras_size
        
        # for supervise
        self.expert_probs_size = expert_probs_size
        self.expert_probs = torch.zeros(num_steps, num_processes, expert_probs_size)
        
        self.hidden_size = hidden_size
        
        
        if self.use_history:
            # 全景特征存储 (不再使用hist_len维度)
            # views = 12 (全景视角数), image_feat_size = 768, angle_feat_size = 2
            self.image_feat_size = 768
            self.angle_feat_size = 2
            self.num_views = 12
            self.pano_img_feats = torch.zeros((num_steps + 1, num_processes, self.num_views, self.image_feat_size))
            self.pano_ang_feats = torch.zeros((num_steps + 1, num_processes, self.num_views, self.angle_feat_size))
    def reset(self):
        super(GlobalRolloutStorage, self).reset()
        if hasattr(self, 'expert_probs'):
            self.expert_probs = torch.zeros_like(self.expert_probs)
        if self.use_history:
            # 重置全景特征
            self.pano_img_feats = torch.zeros_like(self.pano_img_feats)
            self.pano_ang_feats = torch.zeros_like(self.pano_ang_feats)

    def to(self, device):
        super(GlobalRolloutStorage, self).to(device)
        if self.use_history:
            self.pano_img_feats = self.pano_img_feats.to(device)
            self.pano_ang_feats = self.pano_ang_feats.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras, expert_probs=None, 
               pano_img_feats=None, pano_ang_feats=None):
        self.extras[self.step + 1].copy_(extras)
        if expert_probs is not None:
            self.expert_probs[self.step].copy_(expert_probs)
        if self.use_history:
            # 保存全景特征
            if pano_img_feats is not None:
                self.pano_img_feats[self.step + 1].copy_(pano_img_feats)
            if pano_ang_feats is not None:
                self.pano_ang_feats[self.step + 1].copy_(pano_ang_feats)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)

    def after_update(self):
        super(GlobalRolloutStorage, self).after_update()
        if self.use_history:
            # 更新全景特征
            self.pano_img_feats[0].copy_(self.pano_img_feats[-1])
            self.pano_ang_feats[0].copy_(self.pano_ang_feats[-1])
        
    def get_pano_feats(self, step):
        """获取指定 step 的全景特征"""
        return self.pano_img_feats[step], self.pano_ang_feats[step]
        
    def get_actions(self, step):
        """获取历史动作"""
        return self.actions[:step+1] if step > 0 else None
        
    def get_all_pano_feats(self):
        """获取从0到step的所有全景特征作为历史特征"""
        # 返回 (num_processes, step+1, num_views, image_feat_size)
        return self.pano_img_feats[:self.step], self.pano_ang_feats[:self.step]
