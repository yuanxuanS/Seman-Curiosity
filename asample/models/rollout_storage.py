#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict

class GraphRolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        # observation_space,
        action_space,
        # recurrent_hidden_state_size,
        # num_recurrent_layers=1,
        is_double_buffered: bool = False,
    ):
        self.buffers = TensorDict()
        # self.buffers["observations"] = TensorDict()

        ghost_node_max_num = 200
        ghost_node_embeds_size = 768
        ghost_pos_size = 7
        self.buffers['gmap_img_fts'] = torch.from_numpy(
            np.zeros((numsteps + 1, num_envs, ghost_node_max_num, ghost_node_embeds_size), dtype=np.float32)
        )
        self.buffers['gmap_pos_fts'] = torch.from_numpy(
            np.zeros((numsteps + 1, num_envs, ghost_node_max_num, ghost_pos_size), dtype=np.float32)
        )
        self.buffers['gmap_masks'] = torch.from_numpy(
            np.zeros((numsteps + 1, num_envs, ghost_node_max_num), dtype=bool)
        )
        self.gmap_vp_ids = [[] for _ in range(numsteps + 1)]
        # for sensor in observation_space.spaces:
        #     self.buffers["observations"][sensor] = torch.from_numpy(
        #         np.zeros(
        #             (
        #                 numsteps + 1,
        #                 num_envs,
        #                 *observation_space.spaces[sensor].shape,
        #             ),
        #             dtype=np.int32,
        #         )
        #     )

        # self.buffers["recurrent_hidden_states"] = torch.zeros(
        #     numsteps + 1,
        #     num_envs,
        #     num_recurrent_layers,
        #     recurrent_hidden_state_size,
        # )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )
        if action_space.__class__.__name__ == "ActionSpace":
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

        
    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_nav_inputs=None,
        # next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        if next_nav_inputs is not None:
            gmap_vp_ids = next_nav_inputs.pop('gmap_vp_ids')
            gmap_img_fts = next_nav_inputs.pop('gmap_img_fts')
            gmap_pos_fts = next_nav_inputs.pop('gmap_pos_fts')
            gmap_masks = next_nav_inputs.pop('gmap_masks')
            # gmap_img_fts = next_nav_inputs.pop('gmap_img_fts')
            
            next_step = dict(
                gmap_img_fts = gmap_img_fts,
                gmap_pos_fts = gmap_pos_fts,
                gmap_masks = gmap_masks,
                # recurrent_hidden_states=next_recurrent_hidden_states,
                prev_actions=actions,
                masks=next_masks,
            )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )
            self.gmap_vp_ids[self.current_rollout_step_idxs[buffer_index] + 1] = gmap_vp_ids

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]
        self.gmap_vp_ids[0] = self.gmap_vp_ids[self.current_rollout_step_idx]
        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]
        
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )
    
    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            # batch["recurrent_hidden_states"] = batch[
            #     "recurrent_hidden_states"
            # ][0:1]
            t = self.gmap_vp_ids[0 : self.current_rollout_step_idx]
            batch["gmap_vp_ids"] = [[t_[i] for i in inds.tolist()] for t_ in t]
            # yield batch.map(lambda v: v.flatten(0, 1))
            yield batch
