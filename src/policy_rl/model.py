import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from typing import Callable, Optional
from torch import Tensor

import numpy as np

from src.policy_rl.utils.distributions import Categorical, DiagGaussian
from src.policy_rl.utils.model import get_grid, ChannelPool, Flatten, NNBase
from src.policy_rl.envs.utils import depth_utils as du
import cv2
from src.policy_rl.panorama_model import panorama_model, model_config, ModelConfig
from src.policy_rl.panorama_model_non import panorama_model as NonHistoryPanoramaModel
import time
from src.policy_rl.panorama_model import get_all_point_angle_feature

class RBFEncoding(nn.Module):
    def __init__(self, centers, sigma=15.0):
        super(RBFEncoding, self).__init__()
        # centers: 预设的关键步数点，例如 [0, 10, 20, 40, 60, 80]
        self.register_buffer('centers', torch.tensor(centers).float())
        self.sigma = sigma

    def forward(self, d):
        # d: batch * 1
        # 利用广播机制计算: (batch * 1) - (1 * num_centers) -> batch * num_centers
        distances = d - self.centers.unsqueeze(0)
        return torch.exp(-(distances**2) / (2 * self.sigma**2))
    
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
        goal_emb = self.goal_emb(extras[:, 0])

        x = torch.cat((x, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs

class Semantic_map_policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        super(Semantic_map_policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 5, 32, 3, stride=1, padding=1),
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

        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)      # 输入索引，映射为8维向量
        # self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras[:, 0])
        # goal_emb = self.goal_emb(extras[:, 0])

        x = torch.cat((x, orientation_emb), 1)

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


class Uncertainty_Diversity_Policy(NNBase):
    def __init__(self, input_shape, num_actions,
                 recurrent=True, hidden_size=512,
                 num_sem_categories=5, 
                 max_budget=5,
                 input_category=False,
                 input_budget=False,
                 input_sslj=False):
        super(Uncertainty_Diversity_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)
        self.num_sem_categories = num_sem_categories
        self.input_category = input_category
        self.input_budget = input_budget
        self.input_sslj = input_sslj
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

        if input_category:
            # input object statis state
            self.category_encoder = nn.Sequential(
                nn.Linear(num_sem_categories, 64),
                nn.ReLU(),
                nn.Linear(64, 32), # 增加深度，使模型能理解标量间的复杂关系
                nn.ReLU()
            )
        
        if input_budget:
            budget_dim = max_budget + 1
            self.budget_encoder = nn.Sequential(
                nn.Linear(budget_dim, 32),
                nn.ReLU()
            )
        
        # for step encoding
        if input_sslj:
            self.step_encoding = RBFEncoding([10, 20, 40, 60, 80])
            self.sslj_encoder = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU()
            )
        
        # projection layers
        self.extra_dim = 0
        if self.input_category:
            self.extra_dim += 32
        if self.input_budget:
            self.extra_dim += 32
        if self.input_sslj:
            self.extra_dim += 32
        # self.linear1 = nn.Linear(self.conv_output_size + self.extra_dim, hidden_size)
        self.linear1 = nn.Linear(self.extra_dim, hidden_size)
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

    def get_budget_one_hot(self, current_budget, max_budget=5):
        """
        current_budget: 当前剩余的次数 (Tensor, shape: [batch_size])
        max_budget: 最大允许的次数
        return: batch_size, max_budget +1
        """
        # 确保 budget 是长整型，且范围在 [0, max_budget]
        # 注意：one_hot 的类别数通常设为 max_budget + 1，因为包含 0
        current_budget = current_budget.long().clamp(0, max_budget)
        
        # 转换为 one-hot
        # 输出 shape: [batch_size, max_budget + 1]
        one_hot = F.one_hot(current_budget, num_classes=max_budget + 1).float()
        
        return one_hot
    

    
    def forward(self, rgb, rnn_hxs, masks, extras=None):
        resnet_output = self.resnet_l5(rgb[:, :3, :, :])
        conv_output = self.conv(resnet_output)
        
        # extras[:, : num_sem_categories] 是物体个数
        # extras[:, -1] 是动作 budget
        # 注意：物体个数通常需要做简单的归一化(如 /10)或者保持 float
        
        category_info = extras[:, :self.num_sem_categories].float()
        if self.input_category:
            category_info = self.category_encoder(category_info)
        
        budget = extras[:, -2].float()
        budget_vector = self.get_budget_one_hot(budget.T)
        if self.input_budget:
            budget_info = self.budget_encoder(budget_vector)
        
        # 编码距离上一次的步数
        step_since_last_jump = extras[:, -1]
        if self.input_sslj:
            sslj = self.step_encoding(step_since_last_jump.unsqueeze(1))
            sslj = self.sslj_encoder(sslj)
        
        if self.input_category and self.input_budget and self.input_sslj:
            # combined = torch.concat([conv_output.view(  # fnn 1
            #         -1, self.conv_output_size), 
            #                          category_info, 
            #                          budget_info,
            #                          sslj], dim=1)
            combined = torch.concat([ 
                                     category_info, 
                                     budget_info,
                                     sslj
                                     ], dim=1)
        else:
            # TODO
            if self.input_category:
                combined = torch.concat([conv_output.view(  # fnn 1
                    -1, self.conv_output_size), category_info,], dim=1)
            if self.input_budget:
                combined = torch.concat([conv_output.view(  # fnn 1
                    -1, self.conv_output_size), budget_info], dim=1)
            if (not self.input_category) and (not self.input_budget):
                combined = conv_output.view(-1, self.conv_output_size)
                
        x = nn.ReLU()(self.linear1(combined))
        
        # print(f"resnet output: {resnet_output.shape}")
        # print(f"conv output: {conv_output.shape}")
        # print(f"x output: {x.shape}")
        
        if self.dropout > 0:
            x = self.dropout1(x)
        
        x = nn.ReLU()(self.linear2(x))       # fnn 2

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.policy_linear(x))        # action feature
        
        
        # mask invalid tp action
        # --- 计算 Mask ---
        # 初始化全 1 掩码 [Batch, Num_Actions]
        # action_mask = torch.ones_like(budget, )
        # # 找到 budget 为 0 的索引
        invalid_indices = (budget <= 0)
        # # 将这些样本的最后一个动作（索引 -1）设为不可选 (0)
        # action_mask[invalid_indices] = 0
        return self.critic_linear(x).squeeze(-1), x, rnn_hxs, invalid_indices


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        if action_space.__class__.__name__ == "Discrete":
            # num_outputs = action_space.n
            num_outputs = 2
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

        if model_type == 1:
            self.network = Semantic_Curiosity_Policy(
                obs_shape, num_outputs, **base_kwargs)
        elif model_type == 2:
            self.network = Semantic_map_policy(
                obs_shape, **base_kwargs)
        elif model_type == 3:
            self.network = Uncertainty_Diversity_Policy(
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
        self.num_outputs = num_outputs

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
        if self.model_type == 3:
            value, actor_features, rnn_hxs, invalid_indices = self(inputs, rnn_hxs, masks, extras)
            # 将 action_mask 为 0 的位置对应的 Logits 设为极负值
            # 这样在 Softmax 之后，这些动作的概率几乎为 0
            dist = self.dist(actor_features)
            action_mask = torch.ones((dist.logits.shape), device=dist.logits.device)
            action_mask[invalid_indices, -1] = 0
            dist.logits = dist.logits.masked_fill(action_mask == 0, -1e10)
        else:
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
        if self.model_type == 1:
            value, _, _ = self(inputs, rnn_hxs, masks, extras)
        else:
            value, _, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):
        if self.model_type == 3:
            value, actor_features, rnn_hxs,invalid_indices = self(inputs, rnn_hxs, masks, extras)
            dist = self.dist(actor_features)
            action_mask = torch.ones((dist.logits.shape), device=dist.logits.device)
            action_mask[invalid_indices, -1] = 0
            dist.logits = dist.logits.masked_fill(action_mask == 0, -1e10)
        else:
            value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
            dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class RL_Policy2(nn.Module):
    model_config = model_config
    def __init__(self, obs_shape, action_space, device=0,
                 base_kwargs=None, use_history=False,
                 profile_panorama_encoder=False, profile_interval=10):

        super(RL_Policy2, self).__init__()
        
        self.use_history = use_history
        self.device = device
        
        model_config = ModelConfig(**self.model_config)
        model_config.profile_panorama_encoder = profile_panorama_encoder
        model_config.profile_interval = profile_interval
        
        if action_space.__class__.__name__ == "Discrete":
            # num_outputs = action_space.n
            # 根据 use_history 参数选择不同的模型
            if use_history:
                # 使用有历史信息的模型
                self.network = panorama_model(model_config, device)
                num_outputs = 1
            else:
                # 使用无历史信息的模型
                self.network = NonHistoryPanoramaModel(model_config, device)
                num_outputs = 12
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

        if action_space.__class__.__name__ == "Discrete":
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError
    
    @property
    def is_recurrent(self):
        return False
    
    @torch.jit.export
    def forward(self, inputs:Tensor, 
                extras:Optional[Tensor]=None,
                curr_pano_img_feats:Optional[Tensor]=None, 
                curr_pano_ang_feats:Optional[Tensor]=None,
                hist_pano_img_feats:Optional[Tensor]=None, 
                hist_img_feats:Optional[Tensor]=None, 
                hist_masks:Optional[Tensor]=None,
                compute_hist_embed: bool=False):
        """
        Forward 函数: 根据 use_history 参数选择不同的模型调用方式
        
        Args:
            inputs: 当前观测 (全景图) (num_processes, views, H, W, C) 或特征字典
            extras: 额外输入（可选）
            hist_pano_img_feats: 历史全景视角特征
            hist_pano_ang_feats: 历史全景角度特征 (在hist_img_feats参数中)
            hist_actions: 历史动作 (在hist_masks参数中)
            hist_masks: 历史掩码
            compute_hist_embed: 是否计算并返回当前观测的嵌入用于历史存储
        """
        # if self.use_history:
        #     # 历史模型: 使用 forward_with_history 方法
        #     # 如果有原始历史特征，使用新的 forward_with_history 方法
        #     if hist_pano_img_feats is not None:
        #         return self.network.forward_with_history(
        #             curr_pano_img_feats=inputs['pano_img_feats'],
        #             curr_pano_ang_feats=inputs['pano_ang_feats'],
        #             hist_pano_img_feats=hist_pano_img_feats,
        #             hist_pano_ang_feats=hist_img_feats,  # 这里hist_img_feats实际上是hist_pano_ang_feats
        #             hist_actions=hist_masks,  # 这里hist_masks实际上是hist_actions
        #             hist_masks=None,
        #             compute_hist_embed=compute_hist_embed
        #         )
        #     else:
        #         # 无历史时，使用旧的 forward 方法（需要编码 inputs）
        #         # inputs 是原始图像，需要先编码
        #         with torch.no_grad():
        #             ob_img_feats = self.network.encoding(inputs).to(self.device)
                    
        #             ang_feats = get_all_point_angle_feature(self.network.config.angle_feat_size, )
        #             ob_ang_feats = (ang_feats.unsqueeze(0)).repeat(inputs.shape[0], 1, 1).to(self.device)
                
        #         return self.network.forward_with_history(
        #             curr_pano_img_feats=ob_img_feats,
        #             curr_pano_ang_feats=ob_ang_feats,
        #             hist_pano_img_feats=None,
        #             hist_pano_ang_feats=None,
        #             hist_actions=None,
        #             hist_masks=None,
        #             compute_hist_embed=compute_hist_embed
        #         )
        # else:
        # 非历史模型: 使用当前全景图像与角度特征
        return self.network(inputs)
    # @torch.jit.export
    def act(self, inputs:Tensor,
            extras:Optional[Tensor]=None, 
            deterministic: int=False, 
            curr_pano_img_feats:Optional[Tensor]=None, curr_pano_ang_feats:Optional[Tensor]=None,
            hist_pano_img_feats:Optional[Tensor]=None, hist_pano_ang_feats:Optional[Tensor]=None,
            hist_actions:Optional[Tensor]=None, hist_masks:Optional[Tensor]=None,
            compute_hist_embed: int=False):
        """
        Act 函数: 根据 use_history 参数选择不同的模型调用方式
        """
        # if self.use_history:
        #     # 历史模型: 使用 forward_with_history 方法
        #     result = self.network.forward_with_history(
        #         curr_pano_img_feats=curr_pano_img_feats,
        #         curr_pano_ang_feats=curr_pano_ang_feats,
        #         hist_pano_img_feats=hist_pano_img_feats,
        #         hist_pano_ang_feats=hist_pano_ang_feats,
        #         hist_actions=hist_actions,
        #         hist_masks=hist_masks,
        #         compute_hist_embed=compute_hist_embed
        #     )
            
        #     if compute_hist_embed:
        #         value, act_feature, curr_embed = result     # act_feature: env*12*h/2
        #     else:
        #         value, act_feature = result
        # else:
        # 非历史模型: 
        result = self(inputs, None, None, None, None, None, None, False)
        value, action_logits = result
        
        dist = torch.distributions.Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return value, action, action_log_probs
        # return value, action, action_log_probs, dist.probs

    @torch.jit.export
    def get_value(self, inputs:Tensor,
                  extras=None, 
                  curr_pano_img_feats=None, curr_pano_ang_feats=None,
                  hist_pano_img_feats=None, hist_pano_ang_feats=None,
                  hist_actions=None, hist_masks=None):
        """
        Get value: 根据 use_history 参数选择不同的模型调用方式
        """
        # if self.use_history:
        #     # 历史模型
        #     result = self.network.forward_with_history(
        #         curr_pano_img_feats=curr_pano_img_feats,
        #         curr_pano_ang_feats=curr_pano_ang_feats,
        #         hist_pano_img_feats=hist_pano_img_feats,
        #         hist_pano_ang_feats=hist_pano_ang_feats,
        #         hist_actions=hist_actions,
        #         hist_masks=hist_masks,
        #         compute_hist_embed=False
        #     )
        #     value, _ = result
        # else:
        # 非历史模型: 使用当前全景图像与角度特征
        result = self(inputs, None, None, None, None, None, None, False)
        value = result[0]
        return value

    @torch.jit.export
    def evaluate_actions(self, inputs:Tensor,
                         action:Tensor,
                         extras=None, 
                        curr_pano_img_feats=None, curr_pano_ang_feats=None,
                        hist_pano_img_feats=None, hist_pano_ang_feats=None,
                        hist_actions=None, hist_masks=None):
        """
        Evaluate actions: 根据 use_history 参数选择不同的模型调用方式
        """
        # if self.use_history:
        #     # 历史模型
        #     result = self.network.forward_with_history(
        #         curr_pano_img_feats=curr_pano_img_feats,
        #         curr_pano_ang_feats=curr_pano_ang_feats,
        #         hist_pano_img_feats=hist_pano_img_feats,
        #         hist_pano_ang_feats=hist_pano_ang_feats,
        #         hist_actions=hist_actions,
        #         hist_masks=hist_masks,
        #         compute_hist_embed=False
        #     )
            
        #     value, actor_features = result
        # else:
            # 非历史模型: 使用当前全景图像与角度特征
        result = self(inputs, None, None, None, None, None, None, False)
        value, action_logits = result
        
        dist = torch.distributions.Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy
    
    @torch.jit.export
    def evaluate_actions_with_supervise(self, inputs:Tensor,
                                        action:Tensor,
                                        expert_probs:Optional[Tensor]=None,
                                        extras=None, 
                                       curr_pano_img_feats=None, curr_pano_ang_feats=None,
                                       hist_pano_img_feats=None, hist_pano_ang_feats=None,
                                       hist_actions=None, hist_masks=None):
        """
        Evaluate actions with supervise: 根据 use_history 参数选择不同的模型调用方式
        """
        # if self.use_history:
        #     # 历史模型
        #     result = self.network.forward_with_history(
        #         curr_pano_img_feats=curr_pano_img_feats,
        #         curr_pano_ang_feats=curr_pano_ang_feats,
        #         hist_pano_img_feats=hist_pano_img_feats,
        #         hist_pano_ang_feats=hist_pano_ang_feats,
        #         hist_actions=hist_actions,
        #         hist_masks=hist_masks,
        #         compute_hist_embed=False
        #     )
            
        #     value, actor_features = result
        # else:
        # 非历史模型: 使用当前全景图像与角度特征
        result = self(inputs, None, None, None, None, None, None, False)
        value, action_logits = result
        
        dist = torch.distributions.Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy, action_logits
    
class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.det_frame_height
        self.screen_w = args.det_frame_width
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

    def forward(self, obs, pose_obs, maps_last, poses_last, return_curr=False):
        '''
        obs: 0-2: rgb, 3:depth, 4...: semantic
        '''
        bs, c, h, w = obs.size()

        # get geo semantic voxel
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(    # point_cloud_t 维度2：高度
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
        # 以agent为中心的全局地图，更新当前的map
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        # 地图中心，向左偏移视野的一半，得到视野左边界的地图坐标
        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2     
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        # 以中心为agent位置，地图
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs       # 相对t-1的pose1的改变, dx, dy, do (在psoe1坐标系)

        def get_new_pose_batch(pose, rel_pose_change):
            '''
                返回世界坐标系下新的pose
            '''
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)       # 局部位移 (dx, dy)通过旋转矩阵转换到世界坐标系， dy_world
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232     # do_world

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0      # -180, 180范围

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        # 坐标方向 x向右y向下；移动到地图中心，并归一化到 [-1, 1], 坐标系正向反转（affine函数的坐标为 x向左y向上）
        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])       #转为： 向上为x正，，逆时针角度(图像坐标系)增加, 
        
        # 将当前地图进行平移+旋转，和上一时刻地图进行融合
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # update map with last map
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        if return_curr:
            return fp_map_pred, map_pred, pose_pred, current_poses, translated
        else:
            return fp_map_pred, map_pred, pose_pred, current_poses


def inspect_and_catch_undefined_tensor(model: torch.nn.Module):
    """
    无死角扫描模型内部所有的 Parameter 和 Buffer，
    精准揪出引发 strides() 报错的内鬼张量。
    """
    print("🕵️‍♂️ 开始对模型进行全量张量健壮性扫描...")
    has_error = False
    
    # 1. 扫描所有的状态权重 (Parameters 和 Buffers)
    for name, tensor in model.state_dict().items():
        if tensor is None:
            print(f"❌ 发现彻底为 None 的属性: {name}")
            has_error = True
            continue
            
        try:
            # 模拟 C++ 序列化时必调用的底层核心属性
            _ = tensor.stride()
            _ = tensor.storage()
            _ = tensor.data_ptr()
            
            # 特别检查是否为未定义空壳
            if not tensor.is_shared() and tensor.numel() == 0 and tensor.dtype == torch.float32:
                # 有些 0 维张量可能不正常
                pass
                
        except RuntimeError as e:
            print("\n" + "="*60)
            print(f"🚨 【重大嫌疑犯锁定！】")
            print(f"⚠️  张量名称: {name}")
            print(f"⚠️  张量形状 (Shape): {tensor.shape if hasattr(tensor, 'shape') else 'Unknown'}")
            print(f"⚠️  底层错误信息: {str(e)}")
            print("="*60 + "\n")
            has_error = True

    # 2. 扫描模型中可能挂载的普通 Python 属性（有些库喜欢把隐蔽张量挂在 self 下而不注册为 buffer）
    for attr_name in dir(model):
        try:
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                _ = attr.stride()
        except Exception:
            print(f"⚠️ 无法读取模型普通属性的 stride: {attr_name}")
            
    if not has_error:
        print("💡 基础检查未发现异常，嫌疑可能在未被包含进 state_dict 的动态局域变量或 timm 内部复杂子模块中。")

if __name__ == "__main__":
    import gym
    from src.policy_rl.utils.storage import GlobalRolloutStorage

    bs = 1
    
    policy_id = 11  # 1: curiosity, 2: panorama, 11: test curiosity model
    if policy_id == 1:
        observation_space = gym.spaces.Box(0, 255,
                                                    (3, 256,
                                                    256),
                                                    dtype='uint8')
        action_space = gym.spaces.Discrete(3)
        print(f"obs shape:{observation_space.shape}")
        policy = RL_Policy(observation_space.shape, action_space,
                            model_type=1,
                            base_kwargs={'recurrent': 1,
                                        'hidden_size': 256,
                                        'num_sem_categories': 6
                                        })
        checkpoint_path = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/models/curiosity/model_best.pth"
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint)
        policy.eval()
        
        es = 3
        l_rollouts = GlobalRolloutStorage(100,
                                        bs, observation_space.shape,
                                        action_space, policy.rec_state_size,
                                        es)
        
        input = torch.rand(bs, 3, 256, 256,)
        input_rec_state = torch.rand(bs, 256)
        input_masks = torch.ones(bs, dtype=torch.float32)
        
        
        extras = torch.zeros(bs, es)
        local_orientation = torch.zeros(bs, 1).long()
        # locs = full_pose.cpu().numpy()      # 使用全局pose
        # local_orientation[0] = int((locs[0, 2] + 180.0) / 5.)
        # local_xy[0] = torch.from_numpy(locs[0, :2][np.newaxis, :])
        # extras[:, 2] = local_orientation[:, 0]
        # extras[:, :2] = local_xy[:]
        extras = torch.randint(low=0, high=10, size=(bs, es))
        
        l_rollouts.obs[0].copy_(input) 
        l_rollouts.extras[0].copy_(extras)
        
        
        value, action, action_log_probs, rec_states = \
            policy.act(
                l_rollouts.obs[0],
                l_rollouts.rec_states[0],
                l_rollouts.masks[0],
                extras=l_rollouts.extras[0],
                deterministic=False
            )
        action = action.cpu().numpy()
        print(f"✅ 预检查成功！act 函数在提供了全套占位输入后正常执行，输出形状如下：",
          f"value: {value.shape}, action: {action.shape}, action_log_probs: {action_log_probs.shape}")
        
        deterministic = torch.tensor(False, dtype=torch.bool)
        traced_policy_act = torch.jit.trace_module(policy,
                                      {"act":
                                       (l_rollouts.obs[0], 
                                        l_rollouts.rec_states[0],
                                        l_rollouts.masks[0],
                                        l_rollouts.extras[0],
                                        deterministic,
                                        )},
                                      check_trace=True)
        traced_policy_act.save("deployed_rl_policy_curi.pt")
        print("🎉 策略网络已成功打包！生成了 deployed_rl_policy_curi.pt")
    elif policy_id == 11:
        path = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/src/policy_rl/deployed_rl_policy_curi.pt"
        policy = torch.jit.load(str(path))
        policy.eval()
        
        map_size_cm = 2000
        full_pose = torch.zeros(bs, 3).float()
        full_pose[:, :2] = map_size_cm / 100.0 / 2.0     # full pose和local pose单位都是m

        # Helper: apply discrete action to full_pose (in-place)
        def apply_action_to_pose(full_pose_tensor, action_tensor, turn_angle_deg=30.0, step_size_m=0.25):
            """
            full_pose_tensor: Tensor [bs, 3] with columns [x(m), y(m), yaw_deg]
            action_tensor: Tensor [bs] with discrete actions mapping: 0=forward,1=turn_left,2=turn_right
            Updates full_pose_tensor in-place and returns it.
            """
            if not torch.is_tensor(action_tensor):
                action_tensor = torch.tensor(action_tensor)

            action_tensor = action_tensor.to(full_pose_tensor.device)

            # forward mask
            move_mask = (action_tensor == 0)
            left_mask = (action_tensor == 1)
            right_mask = (action_tensor == 2)

            # compute forward displacement only for move steps
            theta_rad = full_pose_tensor[:, 2] * math.pi / 180.0
            dx = step_size_m * torch.cos(theta_rad)
            dy = step_size_m * torch.sin(theta_rad)

            full_pose_tensor[:, 0] = full_pose_tensor[:, 0] + dx * move_mask.to(dtype=full_pose_tensor.dtype)
            full_pose_tensor[:, 1] = full_pose_tensor[:, 1] + dy * move_mask.to(dtype=full_pose_tensor.dtype)

            # update yaw
            full_pose_tensor[:, 2] = full_pose_tensor[:, 2] + turn_angle_deg * left_mask.to(dtype=full_pose_tensor.dtype)
            full_pose_tensor[:, 2] = full_pose_tensor[:, 2] - turn_angle_deg * right_mask.to(dtype=full_pose_tensor.dtype)

            # normalize to [-180, 180)
            full_pose_tensor[:, 2] = torch.fmod(full_pose_tensor[:, 2] + 180.0, 360.0) - 180.0

            return full_pose_tensor

        es = 3
        observation_space = gym.spaces.Box(0, 255,
                                                    (3, 256,
                                                    256),
                                                    dtype='uint8')
        action_space = gym.spaces.Discrete(3)
        l_rollouts = GlobalRolloutStorage(100,
                                        bs, observation_space.shape,
                                        action_space, 256,
                                        es)
        torch.manual_seed(56)
        # use float32 random input to match model expectations (avoid uint8/byte dtype issues)
        input = torch.rand(bs, 3, 256, 256, dtype=torch.float32) * 255.0
        l_rollouts.obs[0].copy_(input)   
        
        extras = torch.zeros(bs, es)
        local_orientation = torch.zeros(bs, 1).long()
        locs = full_pose.cpu().numpy()      # 使用全局pose
        local_orientation[0] = int((locs[0, 2] + 180.0) / 5.)
        local_xy = torch.zeros(bs, 2)
        local_xy[0] = torch.from_numpy(locs[0, :2][np.newaxis, :])
        extras[:, 2] = local_orientation[:, 0]
        extras[:, :2] = local_xy[:]
        # extras = torch.randint(low=0, high=10, size=(bs, es))
        l_rollouts.extras[0].copy_(extras)
        with torch.no_grad():
            value, action, action_log_prob, rec_states  = policy.act(
                l_rollouts.obs[0],
                l_rollouts.rec_states[0],
                l_rollouts.masks[0],
                extras=l_rollouts.extras[0],
                deterministic=torch.tensor(False, dtype=torch.bool)
            )
            print(f"✅ 加载的 JIT 模型 act 函数执行成功，输出如下：",
                f"value: {value}, action: {action}, action_log_prob: {action_log_prob}")
        
        
        reward = torch.rand(bs)  # placeholder reward
        l_masks = torch.ones(bs).float()    
        for step in range(500):
            l_step = step % 100
            
            input = torch.rand(bs, 3, 256, 256, dtype=torch.float32) * 255.0
            # --- Apply discrete action to our internal full_pose estimate ---
            try:
                # action may be a tensor of shape [bs]
                full_pose = apply_action_to_pose(full_pose, action)

                # build extras for next time-step: [x, y, orientation_index]
                # orientation index: map yaw_deg in [-180,180) to [0,71] with 5deg bins
                orient_idx = int((full_pose[0, 2].item() + 180.0) / 5.0)
                orient_idx = max(0, min(71, orient_idx))
                extras_tensor = torch.zeros((bs, es), dtype=torch.float32)
                extras_tensor[:, :2] = full_pose[:, :2]
                extras_tensor[:, 2] = orient_idx

                # write into rollout extras for the next observation used by policy.act
                l_rollouts.extras[l_step + 1].copy_(extras_tensor)
            except Exception:
                # non-fatal: keep running even if pose update fails
                pass
            l_rollouts.insert(
                    input, rec_states,      # state_t+1
                    action, action_log_prob, value,   # action, reward_t
                    reward, l_masks, extras
                )
            with torch.no_grad():
                value, action, action_log_prob, rec_states  = policy.act(
                    l_rollouts.obs[l_step + 1],
                    l_rollouts.rec_states[l_step + 1],
                    l_rollouts.masks[l_step + 1],
                    extras=l_rollouts.extras[l_step + 1],
                    deterministic=torch.tensor(False, dtype=torch.bool)
                )
                print(f"✅ 输出如下：",
                    f"value: {value}, action: {action}, action_log_prob: {action_log_prob}")

            
            reward = torch.rand(bs)
            l_masks = torch.ones(bs).float()
            
            if l_step == 100 - 1:
                l_rollouts.after_update() 
                
                
    elif policy_id == 2:
        observation_space = gym.spaces.Box(0, 255,
                                                    (12, 256,
                                                    256, 3),
                                                    dtype='uint8')
        action_space = gym.spaces.Discrete(12)
        print(f"obs shape:{observation_space.shape}")
    
        policy = RL_Policy2(observation_space.shape,
                    action_space, device='cpu',
                    use_history=False,
                    )
        checkpoint_path = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/models/sequence6-6-1/model_best.pth"
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint)
        policy.eval()
    
    
        input = torch.rand(bs, 12, 256, 256, 3)
        dummy_extras = torch.zeros((bs, 1))

        # 2. 🌟 核心修改：把原本的布尔值 False/True，改用 0 维标量 Tensor 包装
        dummy_deterministic = torch.tensor(False, dtype=torch.bool)

        # 3. 核心全景、历史特征
        dummy_curr_pano_img_feats = torch.randn((bs, 12))
        dummy_curr_pano_ang_feats = torch.randn((bs, 4))

        # 4. 🌟 核心避坑：原代码在这里或者其他地方可能传了 None。
        # 记住，在 Trace 的世界里绝对不能有 None！如果某个特征当前不用，
        # 必须传一个形状对齐、全为 0 的实际 Tensor 进去占位。
        dummy_hist_pano_img_feats = torch.zeros((bs, 10, 12))
        dummy_hist_pano_ang_feats = torch.zeros((bs, 10, 4))
        dummy_hist_actions        = torch.zeros((bs, 10), dtype=torch.long)
        dummy_hist_masks          = torch.ones((bs, 10))

        # 5. 🌟 核心修改：把最后一个布尔值也用 Tensor 包装
        dummy_compute_hist_embed  = torch.tensor(False, dtype=torch.bool)
        
        value, action, action_log_probs = policy.act(input, 
                dummy_extras,
                dummy_deterministic,
                dummy_curr_pano_img_feats,
                dummy_curr_pano_ang_feats,
                dummy_hist_pano_img_feats,
                dummy_hist_pano_ang_feats,
                dummy_hist_actions,
                dummy_hist_masks,
                dummy_compute_hist_embed)
        print(f"✅ 预检查成功！act 函数在提供了全套占位输入后正常执行，输出形状如下：",
            f"value: {value.shape}, action: {action.shape}, action_log_probs: {action_log_probs.shape}")
        traced_policy_act = torch.jit.trace_module(policy,
                                        {"act":
                                        (input, 
                                            dummy_extras,
                                            dummy_deterministic,
                                            dummy_curr_pano_img_feats,
                                            dummy_curr_pano_ang_feats,
                                            dummy_hist_pano_img_feats,
                                            dummy_hist_pano_ang_feats,
                                            dummy_hist_actions,
                                            dummy_hist_masks,
                                            dummy_compute_hist_embed
                                            )},
                                        check_trace=True)
        traced_policy_act.save("deployed_rl_policy.pt")
        print("🎉 策略网络已成功打包！生成了 deployed_rl_policy.pt")
        
        inspect_and_catch_undefined_tensor(policy)
    
    
