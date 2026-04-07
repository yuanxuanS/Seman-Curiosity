import numpy as np
import gym
import habitat
import quaternion
from ..utils import pose as pu
from src.finetune.dataset_utils import save_obs
import os
from src.constants import coco_categories
import bz2
import _pickle as cPickle
import skimage.morphology
from ..utils.fmm_planner import FMMPlanner
import json
import gzip
from src.vqf_constants import target_coco_categories
from habitat_sim.utils.common import quat_from_angle_axis
import math
import pickle
import cv2
import queue

class Transport_Env(habitat.RLEnv):
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
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)
        dataset_info_file = self.episodes_dir + \
            "{split}_info.pbz2".format(split=self.split)
        with bz2.BZ2File(dataset_info_file, 'rb') as f:
            self.dataset_info = cPickle.load(f)
            
        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(4)

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
        
        # for sample locs
        # object loc
        categories = list(target_coco_categories.keys())
        category_objects = {name: [] for name in categories}
        objs = self.habitat_env.sim.semantic_scene.objects[1:]
        for obj in objs:
            if obj.category.name() in categories:
                category_objects[obj.category.name()].append([int(obj.id.replace("_", '')), obj.aabb.center])
        self.category_objects = category_objects
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        
        scene_info = self.dataset_info[scene_name]
        floor_idx = np.random.randint(len(scene_info.keys()))   # 楼层
        sem_map = scene_info[floor_idx]['sem_map']
        self.sample_pts_num = int(sem_map[0].sum() / 5)
        
        if args.sample_mode:
            saved_file = "./data_scene/visibles/info/cate_objs_"+scene_name
            with open(saved_file+".pkl", "wb") as f:
                pickle.dump(category_objects, f)
        
        # for transport action
        if not args.sample_mode:
            tp_loc_f = "./data_scene/visibles/"+scene_name+"_tploc.json"
            with open(tp_loc_f, "r") as f:
                self.tp_loc = json.load(f)       # dict: objid, loc
            
        self.tp_budget = 5
        
        
    def reset(self):
        """Resets the environment to a new episode.
                reset traversible initial location
        
        """
        new_scene = True
        # self.episode_no % self.args.num_train_episodes == 0
        # Initializations
        self.timestep = 0
        self.episode_no += 1

        if new_scene:
            obs = super().reset()
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        
        if self.split == "val":
            if not self.args.sample_mode:
                obs = self.load_episode_loc()       # load episode for inital start position
            else:
                self.sample_obj_visible_loc()
        else:
            if not self.args.sample_mode:
                obs = self.initial_possible_loc()       # train时，随机生成初始位置
            else:
                self.sample_obj_visible_loc()

        tp_loc = self.tp_loc
        self.q = queue.Queue(maxsize=len(tp_loc))
        for i in range(len(tp_loc)):
            # 2. 存入数据
            self.q.put(list(tp_loc.values())[i])
            
        self.init_agent_loc = self.get_sim_location()

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = None
        self.this_sim_location = self.get_sim_location()
        print(f"initial pose: {self.this_sim_location[0]}, {self.this_sim_location[1]}, {self.this_sim_location[2]}")
        self.last_sim_location_z = None
        self.this_sim_location_z, self.this_sim_rot = self.get_sim_location_z()
        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['semantic_gt'] = obs['semantic']
        self.info['depth'] = depth
        self.info['diver_reward'] = 0.
        
        
        self.curr_category_obj_id = {name:[] for name in sorted(list(target_coco_categories.keys()))}
        self.cumu_detected_category = {name:0 for name in sorted(list(target_coco_categories.keys()))}
        
        # for diverisity reward
        if self.args.use_diversity_reward:
            self.info['bbsgt'] = obs['bbsgt']
            self.found_class = []
            self.found_id = []
        # for topo reward
        self.info['position'] = self.this_sim_location[:2]
        
        return state, self.info
    
    def get_navigable_points(self):
        # 得到场景中随机的2w个可移动点
        # navigable_points = np.array([0,self.args.height,0])
        navigable_points = np.array([0,0,0])
        for i in range(self.sample_pts_num):
            navigable_points = np.vstack((navigable_points,self.habitat_env.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
    def sample_obj_visible_loc(self):
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        
        self.nav_pts = self.get_navigable_points()
        all_obj_ids = []
        for cate, values in self.category_objects.items():
            for v in values:
                all_obj_ids.append(v[0])
        surrounding_locs = {id: [] for id in all_obj_ids}
        
        agent_state = self._env.sim.get_agent_state(0)

        valid_rgb_cnt = 0
        for i in range(len(self.nav_pts)):
            pts = self.nav_pts[i]
            pts[1] = 0.     # 不用加高度，Pose sensor类会加上
            agent_state.position = pts
            
            for angle in range(0, 360, 30):
                angle_rad = angle * math.pi/ 180
                quat_yaw = quat_from_angle_axis(angle_rad, np.array([0, 1.0, 0]))  # 绕z轴旋转角度

                agent_state.rotation = quat_yaw
                self._env.sim.set_agent_state(pts, quat_yaw)
                obs = self._env.sim.get_observations_at(pts, quat_yaw)
                obs.update(
                            self._env.task.sensor_suite.get_observations(
                                observations=obs,
                                episode=self._env.current_episode,
                                action={'action': 0, 'action_args':{}},
                                task=self._env.task,
                        ))
                
                
                # if object id in view
                saved = False       # curr obs is saved
                for view_id in list(np.unique(obs['semantic'])):
                    if view_id in all_obj_ids:
                        surrounding_locs[view_id].append(valid_rgb_cnt)
                        
                        # save
                        if not saved:
                            self.save_data(obs, self.rank, 0, valid_rgb_cnt,
                                           data_dir=self.args.sampled_dir+"/"+scene_name)
                            # valid_rgb_cnt += 1
                            saved = True
                if saved:
                    valid_rgb_cnt += 1
        
        print(f"valid rgb :{valid_rgb_cnt}")
        file = "./data_scene/visibles/info/"
        with open(file+scene_name+".json", "w") as f:
            json.dump(surrounding_locs, f)
                
    def load_episode_loc(self):
        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        scene_info = self.dataset_info[scene_name]
        floor_idx = np.random.randint(len(scene_info.keys()))   # 楼层
        self.map_obj_origin = scene_info[floor_idx]['origin']
        
        if self.scene_path != self.last_scene_path: # 如果reset时加载新的环境
            episodes_file = self.episodes_dir + \
                "content/{}_episodes.json.gz".format(scene_name)

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, 'r') as f:
                self.eps_data = json.loads(
                    f.read().decode('utf-8'))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path
            
        # Load episode info
        episode = self.eps_data[self.eps_data_idx]      # episode结束后重新reset，加载数据中不同epsiode的初始位置
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = quaternion.from_float_array(episode["start_rotation"])
        
        self._env.sim.set_agent_state(pos, rot)
        obs = self._env.sim.get_observations_at(pos, rot)
        obs.update(
                self._env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self._env.current_episode,
                    action={'action': 0, 'action_args':{}},
                    task=self._env.task,
            ))
        
        # for transport action
        
        return obs
    
    def initial_possible_loc(self):
        args = self.args
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        scene_info = self.dataset_info[scene_name]
        map_resolution = args.map_resolution

        floor_idx = np.random.randint(len(scene_info.keys()))   # 楼层
        floor_height = scene_info[floor_idx]['floor_height']
        sem_map = scene_info[floor_idx]['sem_map']      # 16*w*h, 一共15类别，0通道是others/背景
        map_obj_origin = scene_info[floor_idx]['origin']

        cat_counts = sem_map.sum(2).sum(1)
        possible_cats = list(np.arange(6))      # 0-5类别
        
        for i in range(6):
            if cat_counts[i + 1] == 0:      # 从0-5的类别中，如果有一个类别的数量为0，则去除这个类别
                possible_cats.remove(i)

        object_boundary = args.success_dist # TODO：
        
        loc_found = False
        while not loc_found:    # 得到合适的目标物体：到该目标的距离可行
            if len(possible_cats) == 0:
                print("No valid objects for {}".format(floor_height))
                eps = eps - 1
                continue
            
            goal_idx = np.random.choice(possible_cats)

            for key, value in coco_categories.items():      # 找到目标的类别名
                if value == goal_idx:
                    goal_name = key

            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(
                sem_map[0], selem) != True      # obstacles: 3W/4W
            traversible = 1 - traversible   # free space: <1W/4w
            
            planner = FMMPlanner(traversible)

            selem = skimage.morphology.disk(
                int(object_boundary * 100. / map_resolution))
            goal_map = skimage.morphology.binary_dilation(
                sem_map[goal_idx + 1], selem) != True       
            goal_map = 1 - goal_map     # 目标物体的地图
            
            planner.set_multi_goal(goal_map)

            m1 = sem_map[0] > 0     # free space
            m2 = planner.fmm_dist > (object_boundary - object_boundary) * 20.0  #      
            m3 = planner.fmm_dist < (20 - object_boundary) * 20.0

            possible_starting_locs = np.logical_and(m1, m2)     # 在距离目标在一定距离范围内，选初始位置
            possible_starting_locs = np.logical_and(
                possible_starting_locs, m3) * 1.
            if possible_starting_locs.sum() != 0:
                loc_found = True
            else:
                print("Invalid object: {} / {} / {}".format(
                    scene_name, floor_height, goal_name))
                possible_cats.remove(goal_idx)
                scene_info[floor_idx]["sem_map"][goal_idx + 1, :, :] = 0.
                self.dataset_info[scene_name][floor_idx][
                    "sem_map"][goal_idx + 1, :, :] = 0.
        
        loc_found = False       # 再找合适的初始位置
        while not loc_found:        
            pos = self._env.sim.sample_navigable_point()
            x = -pos[2]
            y = -pos[0]
            min_x, min_y = map_obj_origin / 100.0   # 
            map_loc = int((-y - min_y) * 20.), int((-x - min_x) * 20.)
            if abs(pos[1] - floor_height) < args.floor_thr / 100.0 and \
                    possible_starting_locs[map_loc[0], map_loc[1]] == 1:    # 且位置高度不能距离地板太远
                loc_found = True

        agent_state = self._env.sim.get_agent_state(0)
        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)
        rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)
        self._env.sim.set_agent_state(pos, rot)
        obs = self._env.sim.get_observations_at(pos, rot)
        obs.update(
                self._env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self._env.current_episode,
                    action={'action': 0, 'action_args':{}},
                    task=self._env.task,
            ))
        self.map_obj_origin = map_obj_origin
        
        return obs
                


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
        if action["action"] == 3:
            print(f"action is transport in {self.rank}")
            loc = self.q.get()
            self.q.put(loc)
            self.tp_budget -= 1
            action["action_args"] = {"tp_loc": [loc[0], loc[1]]}
            obs, _, done, _ = super().step(action,)
            
            
        else:
            obs, _, done, _ = super().step(action)

        
        dx, dy, do = self.get_pose_change()     # update last_sim_location and this_sim_location
        self.info['sensor_pose'] = [dx, dy, do]
        self.this_sim_location = self.get_sim_location()
        
        # save samples(before resize)
        if self.args.save_samples:
            paths = self.save_data(obs)

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep
        self.info['semantic_gt'] = obs['semantic']
        self.info['depth'] = depth
        
        # for diverisity reward
        if self.args.use_diversity_reward:
            self.info['bbsgt'] = obs['bbsgt']
        
        # for topo reward
        self.info['position'] = self.this_sim_location[:2]
        
        # for category object id 
        self.info['category_object'] = [len(self.curr_category_obj_id[name]) for name in sorted(list(target_coco_categories.keys()))]
        # for transport action
        self.info['tp_budget'] =self.tp_budget
        return state, 0., done, self.info

        
    def save_data(self, observations, env=None, episode=None, step=None, data_dir=""):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        data_dir = '{}/episodes_data/'.format(dump_dir) if not data_dir else data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        env = self.rank if env is None else env
        episode = self.episode_no if episode is None else episode
        step = self.timestep if step is None else step
        
        paths = save_obs(data_dir, env, episode, observations, step)
        return paths
    
    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)
    
    def get_reward(self, observations):
        # not used
        return None
    


    def get_done(self, observations, *args):
        if self.info['time'] >= self.args.max_episode_length - 1:       # 
            done = True
        else:
            done = False
        return done
    
    def get_info(self, observations):
        return self.info
    
    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y= sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.), int((-y - min_y) * 20.)
        # o = np.rad2deg(o) + 180.0
        return y, x
    
    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.this_sim_location)  
            # curr_sim_pose, self.last_sim_location)
        # self.last_sim_location = curr_sim_pose
        
        self.last_sim_location = self.this_sim_location
        self.this_sim_location = curr_sim_pose
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
            o -= 2 * np.pi      # 范围放缩到 []
        
        return x, y, o
    
    def get_sim_location_z(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        z = agent_state.position[1]
        rotation = agent_state.rotation
        return z, rotation
    
    def get_action_space(self):
        return self.action_space
    
    def get_obs_space(self):
        return self.observation_space

