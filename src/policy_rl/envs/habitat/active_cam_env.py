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
import random
import pickle
import cv2
import math
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis

class Active_cam_Env(habitat.RLEnv):
    """The Semantic Curiosity environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """
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
        self.action_space = gym.spaces.Discrete(5)

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

        # 随机转移角度
        self.noise_angle = 60
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        self.scene_name = self.scene_path.split("/")[-1].split(".")[0]
        vis_file = f'/home/wpp/Seman-Curiosity/vismap/{self.scene_name}_vismap.txt'
        with open(vis_file, 'rb') as f:
            self.vis_map = pickle.load(f)
    def reset(self):
        """Resets the environment to a new episode.
                reset traversible initial location
        
        """
        new_scene = self.episode_no % self.args.num_train_episodes == 0
        # Initializations
        self.timestep = 0
        # self.episode_no += 1

        if new_scene:
            obs = super().reset()
        
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        
        # 计算vismap时调用
        # self.nav_pts = self.get_navigable_points()
        # self.get_visible_maps()
        
        if self.split == "val":
            obs = self.load_episode_loc()       # load episode for inital start position
        else:
            obs = self.initial_possible_loc()       # train时，随机生成初始位置


        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = None
        self.this_sim_location = self.get_sim_location()
        # print(f"initial pose: {self.this_sim_location[0]}, {self.this_sim_location[1]}, {self.this_sim_location[2]}")
        self.last_sim_location_z = None
        self.this_sim_location_z, self.this_sim_rot = self.get_sim_location_z()
        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['semantic_gt'] = None

        return state, self.info
    
    def get_navigable_points(self):
        # 得到场景中随机的2w个可移动点
        # navigable_points = np.array([0,self.args.height,0])
        navigable_points = np.array([0,0,0])
        for i in range(self.args.num_sample_pts):
            navigable_points = np.vstack((navigable_points,self.habitat_env.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
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
        return obs
    
    def initial_possible_loc(self):
        args = self.args
        
        

        scene_info = self.dataset_info[self.scene_name]
        map_resolution = args.map_resolution

        floor_idx = np.random.randint(len(scene_info.keys()))   # 楼层
        floor_height = scene_info[floor_idx]['floor_height']
        sem_map = scene_info[floor_idx]['sem_map']      # 16*w*h, 一共15类别，0通道是others/背景
        self.map_obj_origin = scene_info[floor_idx]['origin']

        cat_counts = sem_map.sum(2).sum(1)
        possible_cats = target_cls_id_in_scene      # 0-5类别
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
                    break
            
            # 得到可行点
            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(
                sem_map[0], selem) != True      # obstacles: 3W/4W
            traversible = 1 - traversible   # free space: <1W/4w
            
            planner = FMMPlanner(traversible)

            # 在语义地图上得到物体区域
            goal_map_ = sem_map[goal_idx + 1]
            connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
            object_ids = list(np.unique(connected_region[connected_region > 0]))

            # if len(object_ids) > 0:
            #     objects_info[(self.scene_count, episode_id)][pcat]  = {}
            # 如果有多个物体，取其中一个物体
            while not loc_found:
                object_id = random.choice(object_ids)
                
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
                m3 = planner.fmm_dist < (5.5 - object_boundary) * 20.0

                possible_starting_locs = np.logical_and(m1, m2)     # 在距离目标在一定距离范围内，选初始位置
                possible_starting_locs = np.logical_and(
                    possible_starting_locs, m3) * 1.
                if possible_starting_locs.sum() != 0:
                    loc_found = True
                else:
                    print("Invalid object: {} / {} / {} / {}".format(
                        self.scene_name, floor_height, goal_name, object_id))
                    continue
        
                # 过滤能看到物体的位置点
                loc_found = False       
                visible_pts = []
                map_obs = 1-sem_map[0]
                goal_map_one_dil = skimage.morphology.binary_dilation(
                        goal_map_one, selem)
                map_obs[goal_map_one_dil > 0] = 0
                
                # 可视化路径
                # vis_map = np.zeros_like(possible_starting_locs)
                
                vis_map = self.vis_map[self.scene_name][floor_idx][goal_idx][object_id]
                if vis_map.sum() < 10:
                    continue
                
                row_idx, col_idx = np.where(vis_map > 0)
                
                while not loc_found:
                    i = random.choice(range(len(row_idx)))
                    r = row_idx[i]
                    c = col_idx[i]
                    point_real = self.map_coord_to_real((r, c))
                    point = (r, c)
                    
                    # 位置在同一层且可达
                    # if abs(point_real[1] - floor_height) < args.floor_thr / 100.0:    # 不用筛选是否同一层，只要可达即可
                    #     pass
                    # else:
                    #     continue
                    
                    pts_shift = np.array(point_real) - np.array([obj_center_y, obj_center_x])
                    dc = pts_shift[1]     # 
                    dr = pts_shift[0]

                    # 5.5m范围内
                    distances = np.sqrt((dr * dr) + (dc * dc))
                    if distances > 5.5:
                        continue
                    
                    # 移动agent到采样点
                    agent_state = self._env.sim.get_agent_state(0)
                    loc = np.array([point_real[0], floor_height, point_real[1]])   # 设置agent位置
                    agent_state.position = loc
                    
                    # YAW calculation - rotate to object
                    agent_to_obj = np.array(obj_center_real) - agent_state.position
                    agent_local_forward = np.array([0, 0, -1.0]) # y, z, x; 假设agent朝向相机朝向
                    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    flat_to_obj /= flat_dist_to_obj     # agent和object之间的向量归一化

                    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])        # 叉积：得到向量，方向决定了转向
                    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))      # 计算agent的朝向和朝向object的向量之间夹角，计算agent朝向object的角度
                    # 增加一定角度的随机扰动
                    noise_angle = 0 if goal_name == "toilet" else self.noise_angle
                    angle_noise = random.randint(-noise_angle, noise_angle) * math.pi / 180     # -30-30度
                    quat_yaw = quat_from_angle_axis(turn_angle + angle_noise, np.array([0, 1.0, 0]))  # 绕z轴旋转角度
                    # Set agent yaw rotation to look at object
                    agent_state.rotation = quat_yaw
                    # check valid
                    obs = self._env.sim.get_observations_at(loc, quat_yaw)
                    if self.scene_name in self.filtered_scene_obj:
                        obs = self.filter_object(obs, self.filtered_scene_obj[self.scene_name])
                    obs.update(
                        self._env.task.sensor_suite.get_observations(
                            observations=obs,
                            episode=self._env.current_episode,
                            action={'action': 0, 'action_args':{}},
                            task=self._env.task,
                    ))
                    # if object_scene_cnt == 0:
                    valid, self.scene_target_id = self.is_valid_datapoint(obs, goal_name, target_coco_categories)
                    
                    # else:
                    #     valid = self.id_in_view(obs, scene_obj_id)
                        
                    if valid:
                        self.info['goal_name'] = goal_name
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
                
    def is_valid_datapoint(self, observations, category, candidate_cate=None):
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
                    num_occ_pixels = np.where(semantic == id)[0].shape[0]
                    if num_occ_pixels > 0.1 * semantic.shape[-1]*semantic.shape[-1]:
                        return True, int(obj.id[1:])  
                else:
                    if candidate_cate != None:
                        if obj.category.name() in candidate_cate.keys():
                            num_occ_pixels = np.where(semantic == id)[0].shape[0]
                            if num_occ_pixels > 0.1 * semantic.shape[-1]*semantic.shape[-1]:
                                return True, int(obj.id[1:])  
        return False, None
    def map_coord_to_real(self, map_coord):
        map_coord_y, map_coord_x = map_coord
        min_x, min_y = self.map_obj_origin / 100.0
        return map_coord_y / 20. + min_y, map_coord_x / 20. + min_x 
                
                    
    def real_coord_to_map(self, real_coord):
        real_y, real_z, real_x = real_coord
        min_x, min_y = self.map_obj_origin / 100.0
        map_y = int((real_y - min_y) * 20.)     # row
        map_x = int((real_x - min_x) * 20.)
        return (map_y, map_x)
                
    def is_visible_raycasting(self, map_array, map_obstacle, start, objects):
        """
        使用 Raycasting 算法判断从起点是否能看见物体区域。
        
        参数：
        - map_array: 物体的地图, 二维数组, 1表示物体
        - start: 起点坐标 (row, col)
        - objects: 物体区域的坐标集合（例如通过 np.argwhere(map == 1) 得到）
        
        返回：
        - bool: 是否可见
        """
        
        
        # 如果起点本身就是物体，直接可见
        if map_array[start] == 1:
            return True
        
        # 计算物体边界点（周围存在障碍物或地图边界的点）
        boundary_points = Active_cam_Env.find_boundary_points(map_array, objects)
        
        ret = self.check_bdy(map_array, boundary_points, start, map_obstacle)
        return ret

    def check_bdy(self, map_array, boundary_points, start, map_obstacle):
        rows, cols = map_array.shape
        
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
            visible_pts = []
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

    def get_visible_maps(self):
        vis_data = {}
                
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]
        vis_data[scene_name]= {}
        print(f"scene {scene_name}")
        
        scene_info = self.dataset_info[scene_name]

        for floor_idx in list(scene_info.keys()):
            vis_data[scene_name][floor_idx] = {}
            
            floor_height = scene_info[floor_idx]['floor_height']
            sem_map = scene_info[floor_idx]['sem_map']      # 16*w*h, 一共15类别，0通道是others/背景
            self.map_obj_origin = scene_info[floor_idx]['origin']

            cat_counts = sem_map.sum(2).sum(1)
            possible_cats = target_cls_id_in_scene      # 0-5类别
            possible_cats_ = target_cls_id_in_scene.copy()
        
            for i in possible_cats_:
                if cat_counts[i+1] == 0:
                    possible_cats.remove(i)
            
            # 可行地图
            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(
                sem_map[0], selem) != True      # obstacles: 3W/4W
            traversible = 1 - traversible   # free space: <1W/4w
            # planner = FMMPlanner(traversible)
            nav_pts = self.get_all_navi_pts(traversible)
            # nav_pts = self.nav_pts
            
            
            # 障碍地图
            map_obs = 1-sem_map[0]
            
            for goal_idx in possible_cats:
                
                vis_data[scene_name][floor_idx][goal_idx] = {}
                
                for key, value in target_coco_categories.items():      # 找到目标的类别名
                    if value == goal_idx:
                        goal_name = key

                # 在语义地图上得到物体区域
                goal_map_ = sem_map[goal_idx + 1]
                connected_region, num = skimage.morphology.label(goal_map_, connectivity=1, return_num=True)
                object_ids = np.unique(connected_region[connected_region > 0])

                for object_id in object_ids:
                    # vis_data[scene_name][floor_idx][goal_idx][object_id] = None
                    
                    goal_map_one = np.zeros_like(goal_map_)
                    goal_map_one[connected_region == object_id] = 1
                    cv2.imwrite(f"./vismap/{scene_name}_flr_{floor_idx}_cat_{goal_idx}_obj_{object_id}_objmap.png", goal_map_one*255)
                                
                    # 在真实世界坐标下，该物体的中心
                    rows, cols = np.where(goal_map_one > 0)
                    obj_center = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
                    obj_center_y, obj_center_x = self.map_coord_to_real(obj_center)
                    obj_center_real =  obj_center_y, floor_height,  obj_center_x
                    
                    goal_map = goal_map_one
            
                    # planner.set_multi_goal(goal_map)
                    
                    # 膨胀地图
                    goal_map_one_dil = skimage.morphology.binary_dilation(
                            goal_map_one, selem)
                    map_obs[goal_map_one_dil > 0] = 0
                    
                    possible_starting_locs = traversible        # 考虑所有free space
                    # 可视化路径
                    vis_map = np.zeros_like(possible_starting_locs)
                    
                    
                    for i in range(len(nav_pts)):
                        point_real = nav_pts[i]
                        point = self.real_coord_to_map(point_real)
                        
                        # 同一层且可达
                        if not (abs(point_real[1] - floor_height) < self.args.floor_thr / 100.0 and \
                            possible_starting_locs[point[0], point[1]] == 1):
                                continue
                        
                        # raycasting算法检测位置点能看到物体
                        # object_pixels = np.where(goal_map_one > 0)
                        # visible = self.is_visible_raycasting(goal_map_one, map_obs, point, object_pixels)
                        # if visible:
                            # visible_pts.append([point, point_real])
                        agent_state = self._env.sim.get_agent_state(0)
                        loc = np.array([point_real[0], floor_height, point_real[2]])   # 设置agent位置
                        agent_state.position = loc
                            
                        # 视野中有物体
                        # YAW calculation - rotate to object
                        agent_to_obj = np.array(obj_center_real) - agent_state.position
                        agent_local_forward = np.array([0, 0, -1.0]) # y, z, x; 假设agent朝向相机朝向
                        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                        flat_to_obj /= flat_dist_to_obj     # agent和object之间的向量归一化

                        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])        # 叉积：得到向量，方向决定了转向
                        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))      # 计算agent的朝向和朝向object的向量之间夹角，计算agent朝向object的角度
                        # 增加一定角度的随机扰动
                        quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))  # 绕z轴旋转角度
                
                        # check valid
                        obs = self._env.sim.get_observations_at(loc, quat_yaw)
                        if self.scene_name in self.filtered_scene_obj:
                            obs = self.filter_object(obs, self.filtered_scene_obj[self.scene_name])
                        obs.update(
                            self._env.task.sensor_suite.get_observations(
                                observations=obs,
                                episode=self._env.current_episode,
                                action={'action': 0, 'action_args':{}},
                                task=self._env.task,
                        ))
                        valid, self.scene_target_id = self.is_valid_datapoint(obs, goal_name, None)
                
                        if valid:
                            vis_map[point[0], point[1]] = 1
                            
                    vis_data[scene_name][floor_idx][goal_idx][object_id] = vis_map

                    cv2.imwrite(f"./vismap/{scene_name}_flr_{floor_idx}_cat_{goal_idx}_obj_{object_id}_vismap.png", vis_map*255)
        # np.savez(f"./{scene_name}_vismap.npz", data=vis_data)
        
        pth = f"./vismap/{scene_name}_vismap.txt"
        with open(pth, 'wb') as file:
            pickle.dump(vis_data, file)
        
        return  True
    
    def get_all_navi_pts(self, free_map):
        '''get all real naivable loc in free space'''
        navigable_points = np.array([0,0,0])
        
        rows, cols = np.where(free_map > 0)
        for r, c in zip(rows, cols):
            real_y, real_x = self.map_coord_to_real((r, c))
            loc_real =  np.array([real_y, 0,  real_x])  # height is set to 0
            navigable_points = np.vstack((navigable_points, loc_real))
        return navigable_points[1:]
    
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
            os.makedirs(data_dir, exist_ok=True)
        paths = save_obs(data_dir, self.rank, self.episode_no, observations, self.timestep)
        return paths
    
    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)
    
    def get_reward(self, observations):
        # not used
        return None
    


    def get_done(self, observations, action=None):
        '''
        called in super().step()
        '''
        
        if action['action'] == 0:
            return True
        
        return self._env._episode_over
    
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

