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
from src.vqf_constants import target_coco_categories_mapping, target_coco_categories, category_id_maps, target_cls_id_in_scene
import cv2

class Vsqf_v3_Env(habitat.RLEnv):
    """The Vsqf environment class. The class is responsible
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

    def find_closest_obj(self):
        '''
        更新agent和目标类别物体中的最近物体
        '''
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        
        min_dist = 1e4
        for cls_, planners in self.objects_planner_dict.items():
            for planner in planners:
                curr_distance = planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
                if curr_distance < min_dist:
                    min_dist = curr_distance
                    self.nearest_obj_planner = planner
                    self.nearest_obj = cls_
                    self.prev_distance = curr_distance

        return self.nearest_obj_planner, self.nearest_obj
        
    def reset(self):
        """Resets the environment to a new episode.
                reset traversible initial location
        
        """
        new_scene = self.episode_no % self.args.num_train_episodes == 0
        # Initializations
        self.timestep = 0
        self.episode_no += 1

        if new_scene:
            obs = super().reset()
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        
        if self.split == "val":
            obs = self.load_episode_loc()       # load episode for inital start position
        else:
            obs = self.initial_possible_loc()       # train时，随机生成初始位置


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
        
        #vsqf 
        self.info['sample_stage'] = False
        self.info['sample_step'] = 0
        self.info['found_classes'] = []
        
        # 计算最近目标
        self.nearest_obj_planner, self.nearest_obj = self.find_closest_obj()
        return state, self.info
    
    def load_episode_loc(self):
        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

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
        # 记录目标物体语义地图，用于计算和物体距离
        
        scene_info = self.dataset_info[scene_name]
        floor_idx = np.random.randint(len(scene_info.keys()))   # 楼层
        sem_map = scene_info[floor_idx]['sem_map']      # 16*w*h, 一共15类别，0通道是others/背景


        cat_counts = sem_map.sum(2).sum(1)
        possible_cats = target_cls_id_in_scene
        possible_cats_ = target_cls_id_in_scene.copy()
        
        self.objects_planner_dict = {name: [] for name in target_coco_categories.keys()}
        
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
            sem_map[0], selem) != True
        traversible = 1 - traversible
        
        planner = FMMPlanner(traversible)
        
        ## ry.name()].append(obj)
        
        
        for i in possible_cats_:
            if cat_counts[i + 1] == 0:      # 从0-5的类别中，如果有一个类别的数量为0，则去除这个类别
                possible_cats.remove(i)
                
        for pcat in possible_cats:
            goal_idx = pcat
            goal_name = None
            for key, value in target_coco_categories.items():      # 目标类别的名字
                if value == goal_idx:
                    goal_name = key
                    break
            
            # 在语义地图上得到物体区域
            goal_map_ = sem_map[goal_idx + 1]
            connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
            object_ids = list(np.unique(connected_region[connected_region > 0]))
            for object_id in object_ids:
                goal_map_one = np.ones_like(goal_map_)
                goal_map_one[connected_region == object_id] = 0
                
                selem = skimage.morphology.disk(2)
                goal_map_one_ = skimage.morphology.binary_dilation(
                    goal_map_one, selem) != True
                goal_map_one_ = 1 - goal_map_one_
                
                planner.set_multi_goal(goal_map_one_)
                self.objects_planner_dict[goal_name].append(planner)
                
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
        possible_cats = target_cls_id_in_scene
        possible_cats_ = target_cls_id_in_scene.copy()
        
        for i in possible_cats_:
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

            for key, value in target_coco_categories.items():      # 找到目标的类别名
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
        
        self.map_obj_origin = map_obj_origin
        
        # 记录目标物体语义地图，用于计算和物体距离
        self.objects_planner_dict = {name: [] for name in target_coco_categories.keys()}
        
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
            sem_map[0], selem) != True
        traversible = 1 - traversible
        
        planner = FMMPlanner(traversible)
        
        ## 
        # objects_dict = {name: [] for name in target_coco_categories.keys()}
        # for obj in self._env.sim.semantic_scene.objects[1:]:
        #     if obj.category.name() in target_coco_categories.keys():
        #         tgt = obj.aabb.center
        #         dist_ = self._env.sim.geodesic_distance(np.array(pos), np.array(tgt))
        #         if dist_ < 100: # 可达
        #             objects_dict[obj.category.name()].append(obj)
        
        for pcat in possible_cats:
            goal_idx = pcat
            goal_name = None
            for key, value in target_coco_categories.items():      # 目标类别的名字
                if value == goal_idx:
                    goal_name = key
                    break
            
            # 在语义地图上得到物体区域
            goal_map_ = sem_map[goal_idx + 1]
            connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
            object_ids = list(np.unique(connected_region[connected_region > 0]))
            for object_id in object_ids:
                goal_map_one = np.ones_like(goal_map_)
                goal_map_one[connected_region == object_id] = 0
                
                selem = skimage.morphology.disk(2)
                goal_map_one_ = skimage.morphology.binary_dilation(
                    goal_map_one, selem) != True
                goal_map_one_ = 1 - goal_map_one_
                
                planner.set_multi_goal(goal_map_one_)
                self.objects_planner_dict[goal_name].append(planner)
                    
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
        obs, _, done, _ = super().step(action)
 
        # reset location if on floor
        last_sim_location_z = self.this_sim_location_z
        this_sim_location_z, this_sim_rot = self.get_sim_location_z()
        self.info['on_floor'] = (abs(this_sim_location_z - last_sim_location_z) > 0.1)
        if self.info['on_floor']:
            x, y, o = self.this_sim_location    # not update, thus 'this_sim_'
            z = self.this_sim_location_z
            pos = np.array([-y, z, -x])
            self._env.sim.set_agent_state(pos, self.this_sim_rot)
            obs = self._env.sim.get_observations_at(pos, self.this_sim_rot)
            obs.update(
                self._env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self._env.current_episode,
                    action={'action': 0, 'action_args':{}},
                    task=self._env.task,
            ))
        # get newest pose( especially after checking if on floor)
        # self.last_sim_location = self.this_sim_location
        # self.this_sim_location = self.get_sim_location()
        self.last_sim_location_z = self.this_sim_location_z
        self.last_sim_rot = self.this_sim_rot
        self.this_sim_location_z, self.this_sim_rot = self.get_sim_location_z()
        
        dx, dy, do = self.get_pose_change()     # update last_sim_location and this_sim_location
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
            curr_sim_pose, self.this_sim_location)  
            # curr_sim_pose, self.last_sim_location)
        # self.last_sim_location = curr_sim_pose
        
        self.last_sim_location = self.this_sim_location
        self.this_sim_location = curr_sim_pose
        return dx, dy, do
    
    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.), int((-y - min_y) * 20.)

        o = np.rad2deg(o) + 180.0
        return y, x, o
    
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

