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
import cv2
import networkx as nx
import random
import math
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis
from detectron2.utils.visualizer import ColorMode, Visualizer
import pickle
from src.vqf_constants import target_classes, target_cls_id_in_scene, target_coco_categories

class Sample_Obj_Env(habitat.RLEnv):
    '''
    For object sample collection. For target object, sample from multi distance and angle 
    '''
    filtered_scene_obj = {"Collierville": [42, 38], "Darden": [44, 48, 21,32,31 ], "Markleeville": [22, 27],
                          "Wiconisco":[30, 36]}
    
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
                                                (4, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')
        
        self.included_classes = target_classes #'potted plant']
        self.num_views = 25
        self.verbose = False
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
        
        # save dir
        output_path = "/data/wpp_data/data/rand_test/"
        os.makedirs(output_path, exist_ok=True)
        data_pth = output_path+"/data/"
        os.makedirs(data_pth, exist_ok=True)
        img_pth = output_path+"/imgs/"
        os.makedirs(img_pth, exist_ok=True)
        info_path = output_path
        
        self.scene_count = 0       # 环境id，从0开始，每新环境+1
        
        self.nav_pts = self.get_navigable_points()
        if self.split == "val":
            obs = self.sample_possible_loc(data_pth, img_pth, info_path)       
            obs = self.sample_possible_loc_rand(data_pth, img_pth, info_path)       
        else:
            obs = self.sample_possible_loc(data_pth, img_pth, info_path)       


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

        self.instance_id_to_name = {int(obj.id.split("_")[-1]): obj.category.name()
                                   for obj in self._env.sim.semantic_scene.objects
                                   if obj != None}
        return state, self.info
    
    
    def get_navigable_points(self):
        # 得到场景中随机的2w个可移动点
        # navigable_points = np.array([0,self.args.height,0])
        navigable_points = np.array([0,0,0])
        for i in range(self.args.num_sample_pts):
            navigable_points = np.vstack((navigable_points,self.habitat_env.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
    # def get_navigable_points_near(self, center, radius):
    #     # 得到场景中随机的2w个可移动点
    #     navigable_points = np.array([0,0,0])
    #     for i in range(20000):
    #         navigable_points = np.vstack((navigable_points,self.habitat_env.sim.pathfinder.get_random_navigable_point_near(center, radius)))
    #     return navigable_points
    

    
    
    def sample_possible_loc(self, data_pth, img_pth, info_path):
        args = self.args
        self.scene_objs = [obj_id for obj_id in range(len(self.habitat_env.sim.semantic_scene.objects))]
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        if self.scene_path != self.last_scene_path: # 如果reset时加载新的环境
            self.scene_count += 1
            self.last_scene_path = self.scene_path
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        # if scene_name == "Collierville":
        #     pass
        
        scene_info = self.dataset_info[scene_name]
        print(f"scene_{scene_name} has floor: {scene_info.keys()}")
        map_resolution = args.map_resolution

        episode_id = 0  # 每一楼层同一个episode
        
        step_id = 0 # 每次采样一个object为一个step
        objects_info = {}
        
        for floor_idx in list(scene_info.keys()):
            objects_info[(self.scene_count, episode_id)] = {}
            
            step_id = 0

            floor_height = scene_info[floor_idx]['floor_height']
            sem_map = scene_info[floor_idx]['sem_map']      # 16*w*h, 一共15类别，0通道是others/背景
            self.map_obj_origin = scene_info[floor_idx]['origin']

            # 取语义地图的前6个类别
            cat_counts = sem_map.sum(2).sum(1)
            possible_cats = target_cls_id_in_scene      # 目标类别索引 
            possible_cats_ = target_cls_id_in_scene
            
            for i in possible_cats_:
                if cat_counts[i + 1] == 0:      # 如某类别没有物体，则去除这个类别
                    possible_cats.remove(i)

            object_boundary = args.success_dist 
            assert len(possible_cats) > 0, "No valid objects for {}".format(floor_height)
            # 找目标物体：可达物体
        
            for pcat in possible_cats:
                
                goal_idx = pcat
                goal_name = None
                for key, value in target_coco_categories.items():      # 目标类别的名字
                    if value == goal_idx:
                        goal_name = key
                        break

                # 该目标物体可达
                selem = skimage.morphology.disk(2)
                traversible = skimage.morphology.binary_dilation(
                    sem_map[0], selem) != True      # obstacles: 3W/4W
                traversible = 1 - traversible   # free space: <1W/4w
                
                planner = FMMPlanner(traversible)
            
                # 在语义地图上得到物体区域
                goal_map_ = sem_map[goal_idx + 1]
                connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
                object_ids = list(np.unique(connected_region[connected_region > 0]))

                if len(object_ids) > 0:
                    objects_info[(self.scene_count, episode_id)][pcat]  = {}
                # 如果有多个物体，取其中一个物体
                for object_id in object_ids:
                    
                    goal_map_one = np.zeros_like(goal_map_)
                    goal_map_one[connected_region == object_id] = 1
                
                    # 得到：在真实世界坐标下，该物体的中心
                    rows, cols = np.where(goal_map_one > 0)
                    obj_center = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
                    obj_center_y, obj_center_x = self.map_coord_to_real(obj_center)
                    obj_center_real =  obj_center_y, floor_height,  obj_center_x

                    selem = skimage.morphology.disk(
                        int(object_boundary * 100. / map_resolution))
                    
                    # goal_map = skimage.morphology.binary_dilation(
                    #     goal_map_one, selem) != True       
                    # goal_map = 1 - goal_map     # 目标物体的地图
                    goal_map = goal_map_one
                
                    planner.set_multi_goal(goal_map)
                    # 选择距离目标一定范围的可行点
                    m1 = sem_map[0] > 0     # free space
                    m2 = planner.fmm_dist > (object_boundary - object_boundary) * 20.0  #      
                    m3 = planner.fmm_dist < (20 - object_boundary) * 20.0

                    possible_starting_locs = np.logical_and(m1, m2)     # 在距离目标在一定距离范围内，选初始位置
                    possible_starting_locs = np.logical_and(
                        possible_starting_locs, m3) * 1.
                    if possible_starting_locs.sum() != 0:
                        loc_found = True
                    else:
                        print("Invalid object: {} / {} / {} / {}".format(
                            scene_name, floor_height, goal_name, object_id))
                        continue
            
                    # 过滤能看到物体的位置点
                    loc_found = False       
                    visible_pts = []
                    map_obs = 1-sem_map[0]
                    goal_map_one_dil = skimage.morphology.binary_dilation(
                            goal_map_one, selem)
                    map_obs[goal_map_one_dil > 0] = 0
                    # 可视化路径
                    vis_map = np.zeros_like(possible_starting_locs)
                    for i in range(len(self.nav_pts)):
                        point_real = self.nav_pts[i]

                        point = self.real_coord_to_map(point_real)
                        
                        # 同一层且可达
                        if abs(point_real[1] - floor_height) < args.floor_thr / 100.0 and \
                             possible_starting_locs[point[0], point[1]] == 1:    # 不用筛选是否同一层，只要可达即可
                            pass
                        else:
                            continue
                        
                        #过滤位置点： raycasting算法检测能看到物体的
                        object_pixels = np.where(goal_map_one > 0)
                        visible = Sample_Obj_Env.is_visible_raycasting(goal_map_one, map_obs, point, object_pixels)
                        if visible:
                            visible_pts.append([point, point_real])
                            vis_map[point[0], point[1]] = 1
                            # break
                    
                    if len(visible_pts) == 0:
                        print(f"no valid visible points")
                        continue
                    else:
                        objects_info[(self.scene_count, episode_id)][pcat][object_id] = None
                        
                    # 根据角度、距离划分区间
                    visible_pts_real = np.concatenate([p[1][None, ...] for p in visible_pts], axis=0)
                    visible_pts_shift = visible_pts_real - obj_center_real
                    
                    dz = visible_pts_shift[:,2]     # 
                    dx = visible_pts_shift[:,0]
                    dy = visible_pts_shift[:,1]
                    
                    # yaw为agent指向物体的向量和世界正向(0,0,1)的夹角；如果yaw=0，正好向量一致； 则物体朝向为(0,0,-1)
                    valid_yaw = np.degrees(np.arctan2(dz, dx))
                    nbins = 36
                    bins = np.linspace(-180, 180, nbins+1)
                    bin_yaw = np.digitize(valid_yaw, bins)      # 分割成18个离散角度，每个点的对应区间中
                    
                    
                    # 距离区间
                    distances = np.sqrt((dx * dx) + (dz * dz))
                    bin_dis = args.sample_pt_distance_interval
                    bins_dis = np.arange(distances.min(), distances.max()+bin_dis, bin_dis)    # 生成距离区间边界
                    hist, bin_edges = np.histogram(distances, bins_dis) # 划分区间，统计每个区间的数值
                    
                    # find valid points to sample 
                    intervals_center = {}       # 所有区间的中点
                    
                    idx = 1
                    for dis_idx in bin_edges[:-1]:
                        dis_edge_max = dis_idx + bin_dis
                        indices_ = (distances >= dis_idx) * (distances < dis_edge_max)     # 距离范围内
                        for yaw_idx in range(nbins):
                            indices = indices_ *(bin_yaw == yaw_idx+1) # 角度范围内
                            pts = visible_pts_real[indices]
                            if len(pts) > 0:
                                intervals_center.setdefault((dis_idx, yaw_idx), []).append(pts.mean(0))
                            idx += 1

                    valid = False
                    candidates = list(intervals_center.keys())
                    if not len(candidates):
                        raise Exception("No valid point can see object!")

                    # save sample of the object
                    object_samples = {}
                    object_scene_id = None
                    object_scene_cnt = 0
                    for cand in candidates:

                        start_loc = intervals_center[cand][0]
                        
                        # 移动agent到采样点
                        agent_state = self._env.sim.get_agent_state(0)
                        agent_state.position = start_loc       

                        # YAW calculation - rotate to object
                        agent_to_obj = np.array(obj_center_real) - agent_state.position
                        agent_local_forward = np.array([0, 0, -1.0]) # y, z, x; 假设agent朝向相机朝向
                        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                        flat_to_obj /= flat_dist_to_obj     # agent和object之间的向量归一化

                        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])        # 叉积：得到向量，方向决定了转向
                        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))      # 计算agent的朝向和朝向object的向量之间夹角，计算agent朝向object的角度
                        quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))  # 绕z轴旋转角度
                        # Set agent yaw rotation to look at object
                        agent_state.rotation = quat_yaw
                        # check valid
                        obs = self._env.sim.get_observations_at(start_loc, quat_yaw)
                        if scene_name in self.filtered_scene_obj:
                            obs = self.filter_object(obs, self.filtered_scene_obj[scene_name])
                        obs.update(
                            self._env.task.sensor_suite.get_observations(
                                observations=obs,
                                episode=self._env.current_episode,
                                action={'action': 0, 'action_args':{}},
                                task=self._env.task,
                        ))
                        if object_scene_cnt == 0:
                            valid, scene_obj_id = self.is_valid_datapoint(obs, goal_name)
                        else:
                            valid = self.id_in_view(obs, scene_obj_id)
                            
                        if valid:  
                            if object_scene_cnt == 0:
                                object_scene_id = scene_obj_id
                                self.scene_objs.remove(object_scene_id)
                                object_scene_cnt += 1
                            
                            step_id += 1
                            self.save_sample(obs, 
                                        data_pth+"/"+scene_name+"/", 
                                        img_pth+"/"+scene_name+"/",
                                        episode_id,
                                        step_id,
                                        goal_name,
                                        object_id,
                                        )   
                            # save: env, episodeid,  object id in semantic; goalname, 
                            # save for this object
                            dis_idx, yaw_idx = cand
                            dis_key = (dis_idx, dis_idx+bin_dis)
                            if dis_key not in object_samples:
                                object_samples[dis_key] = {}
                            object_samples[dis_key][yaw_idx] = step_id
                    objects_info[(self.scene_count, episode_id)][pcat][object_id] = [scene_obj_id, object_samples]                    
                
            episode_id += 1
        
        with open(info_path+"/"+scene_name+"_objects.pkl", 'wb') as f:
            pickle.dump(objects_info, f)
        return obs

    def sample_possible_loc_rand(self, data_pth, img_pth, info_path, max_samples_per_object=100):
        """
        随机选择一个可见位置和方向的简化版本.
        
        与 sample_possible_loc 不同的是，这个函数:
        1. 不按距离和角度区间划分，直接从所有可见点中随机选择
        2. 随机选择朝向(可以是朝向物体，也可以是随机方向)
        
        Args:
            data_pth: 数据保存路径
            img_pth: 图像保存路径
            info_path: 信息保存路径
            max_samples_per_object: 每个物体最多采样的数量
            
        Returns:
            obs: 最后一个有效的观测
        """
        args = self.args
        self.scene_objs = [obj_id for obj_id in range(len(self.habitat_env.sim.semantic_scene.objects))]
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        if self.scene_path != self.last_scene_path:
            self.scene_count += 1
            self.last_scene_path = self.scene_path
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        
        scene_info = self.dataset_info[scene_name]
        print(f"scene_{scene_name} has floor: {scene_info.keys()}")
        map_resolution = args.map_resolution

        episode_id = 0
        step_id = 0
        objects_info = {}
        
        for floor_idx in list(scene_info.keys()):
            objects_info[(self.scene_count, episode_id)] = {}
            step_id = 0

            floor_height = scene_info[floor_idx]['floor_height']
            sem_map = scene_info[floor_idx]['sem_map']
            self.map_obj_origin = scene_info[floor_idx]['origin']

            cat_counts = sem_map.sum(2).sum(1)
            possible_cats = target_cls_id_in_scene
            possible_cats_ = target_cls_id_in_scene.copy()
            
            for i in possible_cats_:
                if cat_counts[i + 1] == 0:
                    possible_cats.remove(i)

            object_boundary = args.success_dist
            assert len(possible_cats) > 0, "No valid objects for {}".format(floor_height)
        
            for pcat in possible_cats:
                goal_idx = pcat
                goal_name = None
                for key, value in target_coco_categories.items():
                    if value == goal_idx:
                        goal_name = key
                        break

                selem = skimage.morphology.disk(2)
                traversible = skimage.morphology.binary_dilation(
                    sem_map[0], selem) != True
                traversible = 1 - traversible
                
                planner = FMMPlanner(traversible)
            
                goal_map_ = sem_map[goal_idx + 1]
                connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
                object_ids = list(np.unique(connected_region[connected_region > 0]))

                if len(object_ids) > 0:
                    objects_info[(self.scene_count, episode_id)][pcat] = {}
                
                for object_id in object_ids:
                    goal_map_one = np.zeros_like(goal_map_)
                    goal_map_one[connected_region == object_id] = 1
                
                    rows, cols = np.where(goal_map_one > 0)
                    obj_center = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
                    obj_center_y, obj_center_x = self.map_coord_to_real(obj_center)
                    obj_center_real = obj_center_y, floor_height, obj_center_x

                    selem = skimage.morphology.disk(
                        int(object_boundary * 100. / map_resolution))
                    goal_map = goal_map_one
                
                    planner.set_multi_goal(goal_map)
                    m1 = sem_map[0] > 0
                    m2 = planner.fmm_dist > (object_boundary - object_boundary) * 20.0
                    m3 = planner.fmm_dist < (20 - object_boundary) * 20.0

                    possible_starting_locs = np.logical_and(m1, m2)
                    possible_starting_locs = np.logical_and(
                        possible_starting_locs, m3) * 1.
                    if possible_starting_locs.sum() == 0:
                        print("Invalid object: {} / {} / {} / {}".format(
                            scene_name, floor_height, goal_name, object_id))
                        continue
            
                    # 收集所有可见点
                    loc_found = False
                    visible_pts = []
                    map_obs = 1 - sem_map[0]
                    goal_map_one_dil = skimage.morphology.binary_dilation(
                            goal_map_one, selem)
                    map_obs[goal_map_one_dil > 0] = 0
                    
                    for i in range(len(self.nav_pts)):
                        point_real = self.nav_pts[i]
                        point = self.real_coord_to_map(point_real)
                        
                        if abs(point_real[1] - floor_height) < args.floor_thr / 100.0 and \
                             possible_starting_locs[point[0], point[1]] == 1:
                            pass
                        else:
                            continue
                        
                        object_pixels = np.where(goal_map_one > 0)
                        visible = Sample_Obj_Env.is_visible_raycasting(goal_map_one, map_obs, point, object_pixels)
                        if visible:
                            visible_pts.append([point, point_real])
                    
                    if len(visible_pts) == 0:
                        print(f"no valid visible points")
                        continue
                    else:
                        objects_info[(self.scene_count, episode_id)][pcat][object_id] = None
                        
                    # 随机采样: 从所有可见点中随机选择
                    visible_pts_real = np.concatenate([p[1][None, ...] for p in visible_pts], axis=0)
                    
                    # 计算每个点到物体的距离和角度(用于朝向物体)
                    visible_pts_shift = visible_pts_real - obj_center_real
                    dz = visible_pts_shift[:, 2]
                    dx = visible_pts_shift[:, 0]
                    distances = np.sqrt((dx * dx) + (dz * dz))
                    yaw_to_obj = np.degrees(np.arctan2(dz, dx))
                    
                    # 随机选择若干个样本
                    num_samples = min(max_samples_per_object, len(visible_pts_real))
                    if num_samples == 0:
                        continue
                    
                    # 随机选择要采样的索引
                    selected_indices = random.sample(range(len(visible_pts_real)), num_samples)
                    
                    object_samples = {}
                    object_scene_id = None
                    object_scene_cnt = 0
                    
                    for idx in selected_indices:
                        start_loc = visible_pts_real[idx]
                        
                        # 移动agent到采样点
                        agent_state = self._env.sim.get_agent_state(0)
                        agent_state.position = start_loc
                        
                        # 随机选择朝向策略
                        rand_dir = random.random()
                        
                        # if rand_dir < 0.5:  # 50%概率朝向物体
                        #     # YAW calculation - rotate to object
                        #     agent_to_obj = np.array(obj_center_real) - agent_state.position
                        #     agent_local_forward = np.array([0, 0, -1.0])
                        #     flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                        #     flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                        #     flat_to_obj /= flat_dist_to_obj

                        #     det = (flat_to_obj[0] * agent_local_forward[2] - agent_local_forward[0] * flat_to_obj[2])
                        #     turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                        #     quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
                        # else:  # 50%概率随机朝向
                        #     # 随机生成一个朝向角
                        random_yaw = random.uniform(0, 2 * np.pi)
                        quat_yaw = quat_from_angle_axis(random_yaw, np.array([0, 1.0, 0]))
                        
                        # 设置agent朝向
                        agent_state.rotation = quat_yaw
                        
                        # 获取观测
                        obs = self._env.sim.get_observations_at(start_loc, quat_yaw)
                        if scene_name in self.filtered_scene_obj:
                            obs = self.filter_object(obs, self.filtered_scene_obj[scene_name])
                        obs.update(
                            self._env.task.sensor_suite.get_observations(
                                observations=obs,
                                episode=self._env.current_episode,
                                action={'action': 0, 'action_args': {}},
                                task=self._env.task,
                        ))
                        
                        if object_scene_cnt == 0:
                            valid, scene_obj_id = self.is_valid_datapoint(obs, goal_name)
                        else:
                            valid = self.id_in_view(obs, scene_obj_id)
                            
                        if valid:
                            if object_scene_cnt == 0:
                                object_scene_id = scene_obj_id
                                self.scene_objs.remove(object_scene_id)
                                object_scene_cnt += 1
                            
                            step_id += 1
                            self.save_sample(obs,
                                        data_pth + "/" + scene_name + "/",
                                        img_pth + "/" + scene_name + "/",
                                        episode_id,
                                        step_id,
                                        goal_name,
                                        object_id,
                                        )
                            
                            # 保存样本信息
                            dist = distances[idx]
                            yaw_idx = int(yaw_to_obj[idx] / 10)  # 简化的区间索引
                            dis_key = (int(dist / 0.5) * 0.5, int(dist / 0.5) * 0.5 + 0.5)
                            if dis_key not in object_samples:
                                object_samples[dis_key] = {}
                            object_samples[dis_key][yaw_idx] = step_id
                    
                    objects_info[(self.scene_count, episode_id)][pcat][object_id] = [object_scene_id, object_samples]
                
            episode_id += 1
        
        # with open(info_path + "/" + scene_name + "_objects.pkl", 'wb') as f:
        #     pickle.dump(objects_info, f)
        return obs

    def filter_object(self, observations, target_obj_ids):
        '''
        target_obj_ids: list of object ids to be filtered
        '''
        
        for sem in ['semantic',]:
            semantic = observations[sem]
            obj_id = np.unique(semantic.astype('uint8'))
            tgt_ids = [id for id in target_obj_ids if id in obj_id]
            if len(tgt_ids) > 0:
                for id in tgt_ids:
                    semantic[semantic == id] = 0        # 只要不是目标类别的id即可
            observations[sem] = semantic
        return observations
        
    def save_sample(self, obs, scene_dir_data, scene_dir_img, epi_id, step_id, goal_name, object_id):
        os.makedirs(scene_dir_data, exist_ok=True)
        os.makedirs(scene_dir_img, exist_ok=True)
        save_obs(scene_dir_data, self.scene_count, epi_id, obs, step_id)
        # save visualized imgs
        img = obs['rgb'][:,:,:3]
        v2 = Visualizer(img)
        v2 = v2.draw_instance_predictions(obs["bbsgt"]["instances"].to("cpu"))                    # potential map
        img = cv2.cvtColor(v2.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(scene_dir_img + "scene_"+str(self.scene_count)+\
            "epi_"+str(epi_id)+\
            "_step_"+str(step_id)+\
                "_goal_"+goal_name+\
                    "_obj"+str(object_id)+".png", img)

    def id_in_view(self, observations, object_id):
        '''
        semantic中有指定目标的mask
        '''
        semantic = observations["semantic"]      # mask的值为对应类别id
        return object_id in list(np.unique(semantic))
    def is_valid_datapoint(self, observations, category):
        '''
        在视野中是否有指定的物体 
        semantic值为object id; 只要id对应的object为目标类别即可；
        '''
        #
        
        semantic = observations["semantic"]      # mask的值为对应类别id
        
        for id in np.unique(semantic):
            if id > 0:
                # 查找对应的object
                obj = self.habitat_env.sim.semantic_scene.objects[id]
                if obj.category.name() == category:
                    if id in self.scene_objs:
                        return True, id
                    else:
                        continue
        # num_occ_pixels = np.where(semantic == main_id)[0].shape[0]
        #print(semantic.shape)
        #print("Number of pixels: ", num_occ_pixels)
        # small_objects = []
        # if mainobj.category.name() in self.small_classes:
        #     if  num_occ_pixels < 0.9*256*256:
        #         return True
        # else:
        #     if num_occ_pixels < 0.9*256*256: 
        #         return True
        return False, None
    
    def plot_traj(self, coord_lst, map):
        color = (0, 255, 0)
        thickness = 3
        arrow_scale = 0.1
        img_scale = 20
        
        traj_vis = np.zeros((map.shape[0]*img_scale, map.shape[1]*img_scale, 3), dtype=np.uint8)
        for i in range(len(coord_lst) - 1):
            start_r, start_c = self.real_coord_to_map(coord_lst[i])
            end_r, end_c = self.real_coord_to_map(coord_lst[i+1])
            cv2.arrowedLine(traj_vis, (start_c*img_scale, start_r*img_scale), (end_c*img_scale, end_r*img_scale), color, thickness)
            
            
        # cv2.imshow("traj", traj_vis*255)
        return
    
    def real_coord_to_map(self, real_coord):
        real_y, real_z, real_x = real_coord
        min_x, min_y = self.map_obj_origin / 100.0
        map_y = int((real_y - min_y) * 20.)     # row
        map_x = int((real_x - min_x) * 20.)
        return (map_y, map_x)
    
    def map_coord_to_real(self, map_coord):
        map_coord_y, map_coord_x = map_coord
        min_x, min_y = self.map_obj_origin / 100.0
        return map_coord_y / 20. + min_y, map_coord_x / 20. + min_x
    
    @staticmethod
    def tsp_fixed_start_dp(coords, start_node):
        '''
        dp[S][i]: 子集S为访问过的点集合, i为点; dp代表访问过的集合,到点i的最短距离
        集合S, 用二进制位表示， 每一位代表对应点是否被访问过
        '''
        n = len(coords)
        dist = np.zeros((n, n))
        # 距离矩阵
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))

        # 初始化DP表
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]  # 记录前驱节点
        dp[1 << start_node][start_node] = 0  # 起始点固定： 仅有起始点的子集-起始点

        # 遍历所有子集
        for mask in range(1 << n):      # 将1的二进制左移n位，相当于乘 2^n次方
            if (mask & (1 << start_node)) == 0:      # 跳过不包含起点的子集
                continue
            for u in range(n):
                if not (mask & (1 << u)):   # u是否在mask子集中？两个二进制数按位运算，如果u不在mask中，结果为全0，否则不为0
                    continue
                for v in range(n):
                    if mask & (1 << v) or u == v:
                        continue
                    new_mask = mask | (1 << v)      # 将v加入子集
                    if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                        dp[new_mask][v] = dp[mask][u] + dist[u][v]
                        parent[new_mask][v] = u
        
        # 找到最短路径（不返回起点）
        final_mask = (1 << n) - 1
        min_dist = float('inf')
        for u in range(n):
            if u == start_node:
                continue
            if dp[final_mask][u] < min_dist:
                min_dist = dp[final_mask][u]
                last_node = u
        
        # 重建路径
        path = []
        mask = final_mask
        while last_node != -1:
            path.append(last_node)
            prev_node = parent[mask][last_node]
            mask ^= (1 << last_node)
            last_node = prev_node
        path.reverse()

        path_coord = [coords[i] for i in path]
        return path_coord

    @staticmethod
    def tsp_greedy_fixed_start(coords, start_node):
        n = len(coords)
        visited = [False] * n
        path = [start_node]
        visited[start_node] = True
        # 距离矩阵
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dis_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        # print(dis_matrix)    
        
        current = start_node
        for _ in range(n - 1):
            next_city = None
            min_dist = float('inf')
            for i in range(n):
                if not visited[i]:
                    # d = np.linalg.norm(coords[current] - coords[i])
                    d = dis_matrix[current, i]
                    if d < min_dist:
                        min_dist = d
                        next_city = i
            path.append(next_city)
            visited[next_city] = True
            current = next_city

        # 返回闭合路径（若需要）
        # path.append(start)
        path_coord = [coords[i] for i in path]
        return path_coord

    @staticmethod
    def solve_tsp_with_start_point(coords, start_node, cycle=False):
        '''
        返回指定起点的最短路径索引
        coords: list[坐标]
        start_node: 指定起点的ind, 一般为0
        '''
        n = len(coords)
        G = nx.Graph()

        # 添加所有边及其权重
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j, weight=np.linalg.norm(coords[i] - coords[j]))

        # Held-Karp 算法（动态规划）
        from networkx.algorithms.approximation import traveling_salesman_problem
        path = traveling_salesman_problem(G, cycle=cycle)
        if cycle:
            path.pop()
        else:
            # 循环直到起始点为端点
            idx = np.where(path ==start_node)
            
        path_coord = [coords[i] for i in path]
        
        return path_coord

    @staticmethod                
    def is_visible_raycasting(map_array, map_obstacle, start, objects):
        """
        使用 Raycasting 算法判断从起点是否能看见物体区域。
        
        参数：
        - map_array: 二维数组，1 表示物体，0 表示障碍物
        - start: 起点坐标 (row, col)
        - objects: 物体区域的坐标集合（例如通过 np.argwhere(map == 1) 得到）
        
        返回：
        - bool: 是否可见
        """
        rows, cols = map_array.shape
        
        # 如果起点本身就是物体，直接可见
        if map_array[start] == 1:
            return True
        
        # 计算物体边界点（周围存在障碍物或地图边界的点）
        boundary_points = Sample_Obj_Env.find_boundary_points(map_array, objects)
        
        # 对每个边界点发射射线
        for boundary_point in boundary_points:
            # 生成射线方向（从起点到边界点）
            direction = (boundary_point[0] - start[0], boundary_point[1] - start[1])
            if direction == (0, 0):  # 跳过起点本身
                continue
            
            # 归一化方向向量（避免射线过长）
            max_steps = max(abs(direction[0]), abs(direction[1])) + 1
            step_x = direction[0] / max_steps
            step_y = direction[1] / max_steps
            
            # 沿射线路径逐步检查
            visible = True
            for i in range(1, max_steps + 1):
                x = start[0] + step_x * i
                y = start[1] + step_y * i
                
                # 检查是否超出地图范围
                if not (0 <= x < rows and 0 <= y < cols):
                    visible = False
                    break
                
                # 检查当前点是否为障碍物
                if map_obstacle[int(x), int(y)] == 1:
                    visible = False
                    break
            
            # 如果射线未被阻挡且到达物体边界，返回可见
            if visible and np.array_equal(np.round([x, y]), boundary_point):
                return True
        
        return False

    @staticmethod
    def find_boundary_points(map_array, objects):
        """找到物体区域的边界点（周围存在障碍物或地图边界的点）"""
        boundary = set()
        rows, cols = map_array.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        
        for (r, c) in zip(objects[0], objects[1]):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # 检查是否越界或邻居为障碍物
                if (nr < 0 or nr >= rows or nc < 0 or nc >= cols or 
                    map_array[nr, nc] == 0):
                    boundary.add((r, c))
                    break
        return list(boundary)

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
    
    def get_sim_location_z(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        z = agent_state.position[1]
        rotation = agent_state.rotation
        return z, rotation
    
    def get_action_space(self):
        return self.action_space
    
    def get_obs_space(self):
        return self.observation_space

