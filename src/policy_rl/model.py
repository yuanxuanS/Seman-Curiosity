import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

import numpy as np

from .utils.distributions import Categorical, DiagGaussian
from .utils.model import get_grid, ChannelPool, Flatten, NNBase
from .envs.utils import depth_utils as du


class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])

        x = torch.cat((x, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs

class Semantic_Curiosity_Policy(NNBase):
    def __init__(self, input_shape, num_actions,
                 recurrent=True, hidden_size=512,
                 num_sem_categories=5):
        super(Semantic_Curiosity_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)
        
        self.dropout = 0.5

        resnet = models.resnet18(pretrained=True)
        # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])

        # Extra convolution layer
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1, 3, input_shape[1], input_shape[2])
        conv_output = self.conv(self.resnet_l5(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        self.linear1 = nn.Linear(self.conv_output_size, hidden_size)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Policy linear layer
        self.policy_linear = nn.Linear(hidden_size, hidden_size // 2)
        # critic linear layer
        self.critic_linear = nn.Linear(hidden_size // 2, 1)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size // 2

    def forward(self, rgb, rnn_hxs, masks, extras=None):
        resnet_output = self.resnet_l5(rgb[:, :3, :, :])
        conv_output = self.conv(resnet_output)
        x = nn.ReLU()(self.linear1(conv_output.view(  # fnn 1
                -1, self.conv_output_size)))
        
        # print(f"resnet output: {resnet_output.shape}")
        # print(f"conv output: {conv_output.shape}")
        # print(f"x output: {x.shape}")
        
        if self.dropout > 0:
            x = self.dropout1(x)
        
        x = nn.ReLU()(self.linear2(x))       # fnn 2

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.policy_linear(x))        # action feature
        return self.critic_linear(x).squeeze(-1), x, rnn_hxs



# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

        if model_type == 1:
            self.network = Semantic_Curiosity_Policy(
                obs_shape, num_outputs, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError
        
        

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent
        

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
            action = action.reshape(-1)
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories
        print(f" semantic num: {self.num_sem_categories}")
        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)       # cls+1, h, w, z
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last):
        '''
        obs: 0-2: rgb, 3:depth, 4...: semantic
        '''
        bs, c, h, w = obs.size()

        # get geo semantic voxel
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()      # 每个像素点的xyz坐标。env, h,w,3 
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.      # 放缩到 [-1,1]
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(      # feat的0通道都是1
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(      # env, num_sem, range_h, range_w, height
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        # get explore and obstable map
        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)
        # seman_ = voxels[:, 1:].max(1)
        # seman_cls_value = seman_.values       # env, cls+1, h, w, z
        # seman_cls = seman_.indices
        # seman_cls[seman_cls_value > 0] += 1     # 避免和free space混合，类别idx从1开始
        
        fp_map_pred = agent_height_proj[:, 0:1, :, :]   # obstacle map
        fp_exp_pred = all_height_proj[:, 0:1, :, :]     # explore map
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        # all map: obstacle, explore, semantics
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # update map with last map
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses


if __name__ == "__main__":
    import gym

    observation_space = gym.spaces.Box(0, 255,
                                                (3, 128,
                                                 128),
                                                dtype='uint8')
    action_space = gym.spaces.Discrete(3)
    print(f"obs shape:{observation_space.shape}")
    
    device = "cuda:1"
    policy = RL_Policy(observation_space.shape,
                action_space, model_type=1,
                base_kwargs={'recurrent': True,
                                      'hidden_size': 512,
                                    #   'num_sem_categories': args.num_sem_categories
                                      })

    bs = 10
    rnn_hxs = torch.rand(bs, 512)
    l_masks = torch.ones(bs)
    extras = None
    input = torch.rand(bs, 3, 128, 128)
    # value, action_feature, rnn_hxs = policy(input, rnn_hxs, l_masks, extras)
    # print(f"shape: \nvalue:{value.shape}\naction feature:{action_feature.shape}\n")

    value, action, action_prob, rnn_hxs = policy.act(input, rnn_hxs, l_masks, extras)
    print(f"value: {value}")
    print(f"action: {action}")
    print(f"action_prob: {action_prob}")
    print(f"rnn_hxs: {rnn_hxs.shape}")
    