import numpy as np


class VectorizedTopologyManager:
    def __init__(self, num_envs, dist_threshold=0.5, check_target=True):
        self.num_envs = num_envs
        self.dist_threshold = dist_threshold
        # 为每个环境维护一个独立的节点列表
        self.env_nodes = [[] for _ in range(num_envs)]
        self.check_target = check_target
        
        self.reward_shape = True
        self.rs_dis_thres1 = 0.2
        self.rs_dis_thres2 = 0.4
        self.rs_reward1 = 0.05
        self.rs_reward2 = 0.2

    def update(self, env_indices, positions, has_targets):
        """
        env_indices: 当前活跃环境的索引列表 (例如 [0, 1, 2, ...])
        positions: 这些环境对应的智能体坐标 [[x1,y1], [x2,y2], ...]
        has_targets: 对应的视野判定布尔值 [True, False, True, ...]
        """
        rewards = np.zeros(len(env_indices))
        
        for i, env_idx in enumerate(env_indices):
            curr_p = np.array(positions[i])
            
            # 1. 视觉约束
            if self.check_target:
                if not has_targets[i]:
                    continue
                
            nodes = self.env_nodes[env_idx]
            
            # 2. 第一个节点判定
            if len(nodes) == 0:
                self.env_nodes[env_idx].append(curr_p)
                rewards[i] = 0.0
                continue
            
            # 3. 全局距离约束 (计算当前点到该环境所有已存节点的距离)
            dists = np.linalg.norm(np.array(nodes) - curr_p, axis=1)
            if np.min(dists) >= self.dist_threshold:
                self.env_nodes[env_idx].append(curr_p)
                rewards[i] = 1.0
            else:
                if self.reward_shape:
                    if np.min(dists) >= self.rs_dis_thres2:
                        rewards[i] = self.rs_reward2
                    elif np.min(dists) >= self.rs_dis_thres1:
                        rewards[i] = self.rs_reward1
                        
                
        return rewards

    def reset_env(self, env_idx):
        """当某个环境结束(Done)并重启时，清空该环境的地图"""
        self.env_nodes[env_idx] = []

import numpy as np


class VectorizedTopologyManagerFeature:
    def __init__(self, num_envs, dist_threshold=3, check_target=True):
        self.num_envs = num_envs
        self.dist_threshold = dist_threshold
        self.w = 0.8
        # 为每个环境维护一个独立的节点列表
        self.env_nodes = [[] for _ in range(num_envs)]
        self.check_target = check_target
        
        self.reward_shape = True
        self.rs_dis_thres1 = self.dist_threshold * 0.2
        self.rs_dis_thres2 = self.dist_threshold * 0.4
        self.rs_reward1 = 0.05
        self.rs_reward2 = 0.2
        
        

    def update(self, env_indices, features, positions, has_targets):
        """
        env_indices: 当前活跃环境的索引列表 (例如 [0, 1, 2, ...])
        features: 环境对应的特征 [[feat1], [feat2] ,...]
        positions: 这些环境对应的智能体坐标 [[x1,y1], [x2,y2], ...]
        has_targets: 对应的视野判定布尔值 [True, False, True, ...]
        """
        rewards = np.zeros(len(env_indices))
        
        for i, env_idx in enumerate(env_indices):
            curr_p = np.array(positions[i])
            curr_feat = np.array(features[i].squeeze(0))
            
            # 1. 视觉约束
            if self.check_target:
                if not has_targets[i]:
                    continue
                
            nodes = self.env_nodes[env_idx]
            
            # 2. 第一个节点判定
            if len(nodes) == 0:
                self.env_nodes[env_idx].append([curr_p, curr_feat])
                rewards[i] = 0.0
                continue
            
            # 3. 全局距离约束 (计算当前点到该环境所有已存节点的距离)
            node_features = np.array([nf[1] for nf in nodes])
            dists = np.linalg.norm(node_features - curr_feat, axis=1)
            print(f" min dists, {np.min(dists)}")
            if np.min(dists) >= self.dist_threshold:
                self.env_nodes[env_idx].append([curr_p, curr_feat])
                rewards[i] = 1.0
                self.dist_threshold =self.w * self.dist_threshold + (1-self.w) * np.min(dists) 
                self.rs_dis_thres1 = self.dist_threshold * 0.2
                self.rs_dis_thres2 = self.dist_threshold * 0.4
            else:
                if self.reward_shape:
                    if np.min(dists) >= self.rs_dis_thres2:
                        rewards[i] = self.rs_reward2
                    elif np.min(dists) >= self.rs_dis_thres1:
                        rewards[i] = self.rs_reward1
                        
                
        return rewards

    def reset_env(self, env_idx):
        """当某个环境结束(Done)并重启时，清空该环境的地图"""
        self.env_nodes[env_idx] = []