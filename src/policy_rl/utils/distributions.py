# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

import torch
import torch.nn as nn

from .model import AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# FixedCategorical = torch.distributions.Categorical

# old_sample = FixedCategorical.sample
# FixedCategorical.sample = lambda self: old_sample(self)

# log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: \
#     log_prob_cat(self, actions.squeeze(-1))
# FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

import torch
from torch.distributions import Categorical

# 🌟 解决方案：显式继承 PyTorch 官方的 Categorical 分布类
# class FixedCategorical(Categorical):
#     """
#     用正规的子类重写和扩展官方分布，彻底淘汰 lambda，完美兼容 TorchScript。
#     """
    
#     # 1. 规避对 sample 的 lambda 绑定
#     # 既然原本的 lambda 只是复读了旧的 sample(self)，直接继承即可，不需要额外重写。
#     # 如果你想显式写出来打点日志，就用标准 def 定义：
#     def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
#         return super().sample(sample_shape)

#     # 2. 🌟 规避原有的 FixedCategorical.log_probs = lambda ...
#     # 改为用标准的 def 函数，并为动作 actions 加上类型注解
#     def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
#         """ 计算对数概率，并自动压平最后一维，对齐框架接口 """
#         # 原 lambda 逻辑：log_prob_cat(self, actions.squeeze(-1))
#         # 在这里直接调用原本继承自父类的 log_prob 算子
#         return self.log_prob(actions.squeeze(-1))

#     # 3. 🌟 规避原有的 FixedCategorical.mode = lambda ...
#     # 改为用 TorchScript 认识的类方法/属性函数
#     def mode(self) -> torch.Tensor:
#         """ 返回概率最大的动作（强化学习中的确定性策略输出） """
#         # 原 lambda 逻辑：self.probs.argmax(dim=1, keepdim=True)
#         return self.probs.argmax(dim=1, keepdim=True)
    
FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: \
    log_prob_normal(self, actions).sum(-1, keepdim=False)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

import torch
import torch.nn.functional as F

class FixedCategorical:
    """
    一个完全脱离 Module 依赖的纯数据/算子动作分布类。
    完美允许在 TorchScript 的 forward 计算流中被随时实例化。
    """
    def __init__(self, logits: torch.Tensor):
        # 统一使用稳定版 log_softmax 计算对数概率
        # 这样连原本复杂的 if-else 条件分支都省去了，让静态图编译器更喜欢
        self.logits = F.log_softmax(logits, dim=-1)

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    # 🌟 显式为方法的返回值加上 -> torch.Tensor 注解，方便静态图编译器识别
    def sample(self) -> torch.Tensor:
        """ 采样动作 """
        return torch.multinomial(self.probs, num_samples=1).squeeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """ 计算选定动作的对数概率 """
        act = actions.long()
        return self.logits.gather(-1, act.unsqueeze(-1)).squeeze(-1)

    def mode(self) -> torch.Tensor:
        """ 确定性输出：直接取最大概率对应的索引 """
        return self.probs.argmax(dim=-1, keepdim=True)
    
    def entropy(self) -> torch.Tensor:
        """ 
        计算离散分布的香农熵 (Shannon Entropy) 
        公式: -sum(p * log(p))
        """
        # 利用元素级相乘，最后在动作维度（最后一维 -1）求和
        # 别忘了公式最前面的负号
        return -(self.probs * self.logits).sum(dim=-1)
    
class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)  # .squeeze(-1)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
