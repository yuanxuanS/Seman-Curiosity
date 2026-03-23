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
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from PIL import Image
import cv2


class Sequence_Env(habitat.RLEnv):
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
        self.action_space = gym.spaces.Discrete(12)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (12, args.env_frame_height,
                                                 args.env_frame_width, 3),
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
        self.frameid=0

        
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
        
        if self.split == "val"  and self.args.eval:
            obs = self.load_episode_loc()       # load episode for inital start position
        else:
            obs = self.initial_possible_loc()       # train时，随机生成初始位置


        self.last_sim_location = None
        self.this_sim_location = self.get_sim_location()
        print(f"initial pose: {self.this_sim_location[0]}, {self.this_sim_location[1]}, {self.this_sim_location[2]}")
        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['sensor_pose_all'] = [[0., 0., 0.]]
        self.info['orient_idx'] = self.map_radians_to_intervals([self.this_sim_location[-1]])
        self.info['no_straight_num'] = 1
        # self.info['semantic_gt'] = obs['semantic']
        # self.info['depth'] = depth
        self.info['reward'] = 0.
        
        obs_type = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 
                    'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210',
                    'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
        panorama_obs = [obs[obt][:,:, ::-1][None, ...] for obt in obs_type]
        self.info['panorama_obs'] = np.concatenate(panorama_obs, axis=0)


        # self.get_navigable_map()
        # 随机采集30个点，测试每个点的周围可行度
        # type=  ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
        # save_dir = "./test_imgs"
        # cnt = 0
        # for i in range(10000):
        #     if cnt >30:
        #         break
        #     loc = self._env.sim.pathfinder.get_random_navigable_point()
        #     loc[1] = 0.03
        #     if not self._env.sim.pathfinder.is_navigable(loc):
        #         print(f"un navigable {i}")
        #         continue
        #     cnt += 1
        #     nav_bool, rotation = self.is_direction_navigable(True)
            
        #     if sum(nav_bool) < 12:
        #         print(f"has un navigable dir {i}")
                
        #     # else:
        #     #     continue
        #     # img
        #     obs = self._env.sim.get_observations_at(loc, rotation)
        #     obs.update(
        #             self._env.task.sensor_suite.get_observations(
        #                 observations=obs,
        #                 episode=self._env.current_episode,
        #                 action={'action': 0, 'action_args':{}},
        #                 task=self._env.task,
        #         ))
        #     # 
        #     processed_imgs = []
        #     for e,t in enumerate(type):
        #         # 转换颜色通道 RGB -> BGR
        #         img = cv2.cvtColor(obs[t], cv2.COLOR_RGB2BGR)
                
        #         # 确定颜色：可行绿色，不可行红色
        #         is_nav = nav_bool[e]
        #         color = (0, 255, 0) if is_nav else (0, 0, 255)
                
        #         # 在子图上绘制文字标注：方向和结果
        #         angle_text = t.replace('rgb_', '') if 'rgb_' in t else '0'
        #         cv2.putText(img, f"Deg: {angle_text}", (10, 30), 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #         cv2.putText(img, f"Nav: {is_nav}", (10, 60), 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
        #         # 绘制边框以区分每张图
        #         cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, 4)
                
        #         processed_imgs.append(img)
            
        #     if len(processed_imgs) == 12:
        #         # 每行 4 张图拼接
        #         row1 = np.hstack(processed_imgs[0:4])
        #         row2 = np.hstack(processed_imgs[4:8])
        #         row3 = np.hstack(processed_imgs[8:12])
                
        #         # 垂直堆叠三行
        #         combined_img = np.vstack([row1, row2, row3])
                
        #         # 在大图上方添加点位信息
        #         final_img = cv2.copyMakeBorder(combined_img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        #         cv2.putText(final_img, f"Point ID: {i}  Pos: {np.round(loc, 2)}", (20, 35), 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        #         # 保存大图
        #         file_name = f"{save_dir}/point_{i}_all_directions.jpg"
        #         cv2.imwrite(file_name, final_img)
            
        return [obs], self.info
    
    def is_direction_navigable(self, get_rot=False):
        '''
        return:list, navigable bool of 12 direction
        '''
        # 1. 获取 Agent 当前状态
        agent_state = self._env.sim.get_agent_state(0)
        agent_pos = agent_state.position
        # 将四元数转换为欧拉角，获取当前的偏航角 (Yaw)
        # Habitat 使用的是右手法则，y轴向上
        rot = quaternion.as_euler_angles(agent_state.rotation)
        current_yaw = rot[1]  # 通常索引1是围绕y轴的旋转

        navigable_list = []
        img_list = []
        radius = 0.25  # 检测半径
        
        # 2. 遍历 12 个方向 (360度 / 12 = 30度步长)
        for i in range(12):
            # 计算当前探测方向的角度 (弧度)
            # 向左转意味着角度增加
            angle = current_yaw + np.deg2rad(i * 30)
            
            # 计算在 XZ 平面上的偏移量
            # 注意：Habitat 中 z 是向前/后，x 是左/右
            dx = radius * np.sin(angle)
            dz = radius * np.cos(angle)
            
            direction_navigable = True
            
            # 3. 在高度上下 0.3m 范围内进行检测
            # 可以根据需要增加采样点，这里检测 [原高度-0.3, 原高度, 原高度+0.3]
            for y_offset in np.arange(agent_pos[1]- 0.3, agent_pos[1] + 0.3, 0.1):
                check_pos = np.array([
                    agent_pos[0] + dx,
                    agent_pos[1] + y_offset,
                    agent_pos[2] + dz
                ], dtype=np.float32)
                
                # 只要有一个高度不可通行，该方向即视为不可导航
                if not self._env.sim.pathfinder.is_navigable(check_pos):
                    direction_navigable = False
                    break
                
            
                
            navigable_list.append(direction_navigable)
        
        if get_rot:
            return navigable_list, agent_state.rotation
        return navigable_list
        
    def get_navigable_map(self):
        '''
        agent_pos: array([x,y,z])
        
        '''
        agent_pos = self._env.sim.get_agent_state(0).position
        agent_height = agent_pos[1]
        bounds_start, bounds_end = self._env.sim.pathfinder.get_bounds()     # same structure as agnet_pos
        bds_x, bds_y, bds_z = bounds_start
        bde_x, bde_y, bde_z = bounds_end
        
        # 定义网格步长
        grid_size = 0.25
        
        # 2. 确定 X 和 Z 方向的采样点个数
        x_range = np.arange(bds_x, bde_x, grid_size)
        z_range = np.arange(bds_z, bde_z, grid_size)
        
        # 3. 遍历高度（从 agent 当前高度开始，按 0.1m 步长向上，这里演示 1.0m 范围）
        # 注意：range 不支持浮点步长，需使用 np.arange
        for agenth in np.arange(agent_pos[1]- 1.1, agent_pos[1] + 1.1, 0.1):
            # 创建一个空白图像 (RGB)
            # 宽度对应 X 轴，高度对应 Z 轴
            img_w, img_h = len(x_range), len(z_range)
            map_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                
            for i, z in enumerate(z_range):
                for j, x in enumerate(x_range):
                    # 构造当前检测点
                    pos = np.array([x, agenth, z], dtype=np.float32)
                    
                    # 4. 检查是否可导航
                    is_nav = self._env.sim.pathfinder.is_navigable(pos)
                    
                    if is_nav:
                        # 可行区域：蓝色 (R=0, G=0, B=255)
                        map_img[i, j] = [0, 0, 255]
                    else:
                        # 不可行区域：红色 (R=255, G=0, B=0)
                        map_img[i, j] = [255, 0, 0]
            
            # 5. 保存地图
            # 将 numpy 数组转为 Image 对象
            img = Image.fromarray(map_img)
            # 文件命名包含高度，保留两位小数防止文件名非法
            save_path = f"./nav_map_height_{agenth:.2f}.png"
            img.save(save_path)
            print(f"Saved: {save_path}")
        
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
        obs.update(
                self._env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self._env.current_episode,
                    action={'action': 0, 'action_args':{}},
                    task=self._env.task,
            ))
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
                
    def turn(self, action, vis_info):    
        ''' angle: 0 ~ 360 degree '''
        act_l = HabitatSimActions.TURN_LEFT     # 2
        act_r = HabitatSimActions.TURN_RIGHT  # 3
        # uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_r].actuation.amount
        # ang_degree = math.degrees(ang)
        # ang_degree = round(ang_degree / uni_l) * uni_l
        

        if action <=6:
            turns = [act_l] * action
        elif action > 6:
            turns = [act_r] * (12 - action)

        observations = []
        dones = []
        sensor_poses = []
        orients = []
        for f, turn in enumerate(turns):
            save = True if f == len(turns) - 1 else False
            obs, done, sp, ort = self.wrap_act(turn, vis_info, save=save)
            observations.append(obs)
            dones.append(done)
            sensor_poses.append(sp)
            orients.append(ort)
        return observations, dones, sensor_poses, orients

    def wrap_act(self, act, vis_info, save=False):
        ''' wrap action, get obs if video_option '''
        
        observations = None
        # if self.video_option:
        #     observations = self._env.step(act)      # TODO
        #     if self.config.SAVE_DATA and save:
        #         rgb = observations['rgb']
        #         rgb = cv2.resize(rgb, (256,256))  
        #         sem_seg_pred, obj = self._get_sem_pred(
        #             rgb.astype(np.uint8), use_seg=True, return_score=False, return_instance=True)
        #         obj = self.filter_instance(obj)
        #         observations['bbspred'] = {'instances':obj}
        #         self.save_data(observations)
                
                
        #     info = self.get_info(observations)
        #     self.video_frames.append(
        #         navigator_video_frame(
        #             observations,
        #             info,
        #             vis_info,
        #         )
        #     )
        # else:
        
        # 转为sim的动作
        # if act == 0: act = HabitatSimActions.MOVE_FORWARD
        # elif act == 1: act = HabitatSimActions.TURN_LEFT
        # elif act == 2: act = HabitatSimActions.TURN_RIGHT
        # self._env.sim.step_without_obs(act)
        # self._env._task.measurements.update_measures(
        #     episode=self._env.current_episode, action=act, task=self._env.task 
        # )
        obs, _, done, _ = super().step({'action': act})
        dx, dy, do = self.get_pose_change()     # update last_sim_location and this_sim_location
        sensor_pose = [dx, dy, do]
        
        
        orient = self.this_sim_location[-1]
        # save samples(before resize)
        if self.args.save_samples:
            if save:
                paths = self.save_data(obs, self.frameid)
            self.frameid += 1
        return obs, done, sensor_pose, orient

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

        # action: 0-6: 
        
        obs, dones, sensor_poses, orients = self.turn(action, None)
        obs_all, dones_all, sensor_poses_all, orients_all = obs, dones, sensor_poses, orients
        self.info['no_straight_num'] = len(obs_all) - 1


        ksteps = 10
        act_f = HabitatSimActions.MOVE_FORWARD      # 1
        for _ in range(ksteps):
            obs, done, sensor_pose, orient = self.wrap_act(act_f, None, save=True)
            obs_all.append(obs)
            dones_all.append(done)
            sensor_poses_all.append(sensor_pose)
            orients_all.append(orient)
            
            # check collision
            collision = False
            tx, ty, _ = self.this_sim_location
            lx, ly, _ = self.last_sim_location
            dist = pu.get_l2_distance(tx, lx, ty, ly)
            collision = dist < self.args.collision_threshold_real
            if collision:
                break

        # self.last_sim_location_z = self.this_sim_location_z
        # self.last_sim_rot = self.this_sim_rot
        # self.this_sim_location_z, self.this_sim_rot = self.get_sim_location_z()
        
        # dx, dy, do = self.get_pose_change()     # update last_sim_location and this_sim_location
        # self.info['sensor_pose'] = [dx, dy, do]
        self.info['sensor_pose_all'] = sensor_poses_all
        self.info['orient_idx'] = self.map_radians_to_intervals(orients_all)
        
        obs_type = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 
                    'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210',
                    'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
        panorama_obs = [obs_all[-1][obt][:,:,::-1][None, ...] for obt in obs_type]
        self.info['panorama_obs'] = np.concatenate(panorama_obs, axis=0)
        
        self.pred_wp_heatmap(obs_all[-1])
        # rgb = obs['rgb'].astype(np.uint8)
        # depth = obs['depth']
        # state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        
        # self.info['time'] = self.timestep
        # self.info['semantic_gt'] = obs['semantic']
        # self.info['depth'] = depth
        
        # return state, 0., done, self.info
        return obs_all, dones_all, self.info
    
    def save_data(self, observations, frameid=0):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        data_dir = '{}/episodes_data/'.format(dump_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        obs_save = {k:v for k,v in observations.items() if k in ['rgb', 'depth', 'semantic', 'gps', 'compass', 'bbsgt', 'position', 'bbspred']}
        # obs_save = observations
        paths = save_obs(data_dir, self.rank, self.episode_no, obs_save, self.timestep, frameid)
        return paths
    
    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)
    
    def get_reward(self, observations):
        # not used
        return None
    


    def get_done(self, observations, *args):
        if self.timestep >= self.args.max_episode_length_straight - 1:       # 
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
            o -= 2 * np.pi      # 范围放缩到 []
        
        return x, y, o
    
    def map_radians_to_intervals(self, rad_array):
        # 1. 将弧度转换为角度 [-180, 180]
        angles = np.degrees(rad_array)
        # print(angles)
        # 2. 处理角度范围，使其与你的区间定义一致
        # 你的区间从 -30 开始，到 330 结束。
        # 我们将所有角度映射到 [-30, 330) 范围内
        angles_shifted = np.mod(angles + 30, 360) - 30
        # print(angles_shifted)
        # 3. 定义区间边界 (左闭右开)
        # 区间 0: [-30, 30), 1: [30, 90), 2: [90, 150), 3: [150, 240), 4: [240, 330)
        # 注意：你只列到了区间 4，剩下的 330 到 -30 实际上是第 6 个区间（区间 5）
        bins = [-30, 30, 90, 150, 240, 330]
        
        # 4. 使用 digitize 进行分类 (返回索引 1-5，减 1 变为 0-4)
        indices = np.digitize(angles_shifted, bins) - 1
        
        # 处理超出 330 的部分归为区间 5 (或者根据你的需求处理)
        indices[indices == 5] = 5 
    
        return list(indices)
    
    def get_sim_location_z(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        z = agent_state.position[1]
        rotation = agent_state.rotation
        return z, rotation
    
    def get_action_space(self):
        return self.action_space
    
    def get_obs_space(self):
        return self.observation_space

