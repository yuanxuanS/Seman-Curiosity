import numpy as np
import gym
import habitat
import quaternion
import envs.utils.pose as pu
from finetune.dataset_utils import save_obs
import os

class Seman_Curio_Env(habitat.RLEnv):
    """The Semantic Curiosity environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """
    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        
        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # episode tracking into
        self.timestep = None
        self.info = {}
        self.last_sim_location = None
        
        # episode id 
        self.episode_no = 0

        
    def reset(self):
        """Resets the environment to a new episode."""

        # Initializations
        self.timestep = 0
        self.episode_no += 1

        obs = super().reset()
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]


        return state, self.info
    
    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # action = action["action"]

        # step
        obs, _, done, _ = super().step(action)

        # TODO: Get pose change?
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]

        # save samples(before resize)
        if self.args.save_samples:
            paths = self.save_data(obs)

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep

        return state, 0., done, self.info
    
    def save_data(self, observations):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        data_dir = '{}/episodes_data/'.format(dump_dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        paths = save_obs(data_dir, self.rank, self.episode_no, observations, self.timestep)
        return paths
    
    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)
    
    def get_reward(self, observations):
        # not used
        return None
    


    def get_done(self, observations):
        if self.info['time'] >= self.args.max_episode_length - 1:       # 
            done = True
        else:
            done = False
        return done
    
    def get_info(self, observations):
        return self.info
    
    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
    
    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        
        return x, y, o
    
    def get_action_space(self):
        return self.action_space
    
    def get_obs_space(self):
        return self.observation_space

