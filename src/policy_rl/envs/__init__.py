import torch

from .habitat import construct_envs
from .sensors import *

def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device) # 环境obs放在主线程的gpu上，保证和训练在同一个gpu
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_reward(self):
        reward = self.venv.get_reward()
        reward = torch.from_numpy(reward).float()
        return reward

    def update_collision_map(self, inputs):
        self.venv.update_collision_map(inputs)
        
    def step_and_preprocess(self, action, wait_env):
        obs, reward, done, info = self.venv.step_and_preprocess(action, wait_env)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info
    
    
    def step_and_pre(self, action, inputs):
        obs, reward, done, info = self.venv.step_and_pre(action, inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_obs_info(self):
        obs_info = self.venv.get_obs_info()
        return obs_info
    
    def save_data(self, obs_info, capture):
        self.venv.save_data(obs_info, capture)
        return
    
    def visualize(self, vis_data):
        self.venv.visualize(vis_data)
        return 
    def close(self):
        return self.venv.close()
    
    def get_action_space(self):
        return self.venv.get_action_space()

    def get_obs_space(self):
        return self.venv.get_obs_space()
