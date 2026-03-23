from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .utils import visualization as vu
from src.constants import color_palette
import os
import torch
from ..envs.utils import pose as pu
from ..envs.habitat.sequence_env import Sequence_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.finetune.dataset_utils import save_obs
import quaternion
from .utils.detect_utils import box_iou_calc
from detectron2.utils.visualizer import ColorMode, Visualizer
from src.vqf_constants import target_coco_categories, \
                            target_coco_categories_mapping, \
                                clsid_name_maps
import time
from src.policy_rl.sequence_utils import (
    get_all_sample_score,
    get_specify_samples,
    aggre_score_in_obj_tracks,
    score_tracks,
    group_by_object_and_score,
)
import math
from collections import Counter
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes, BoxMode
from asample.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from asample.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    ResnetDepthEncoder,
    CLIPEncoder,
)
import gym
import torch.nn.functional as F
from asample.waypoint_pred.utils import nms

class Sequence_Env_Agent(Sequence_Env):
    """The Sem_Curiosity environment agent class. A seperate Sem_Curi_Env_Agent class
    object is used for each environment thread.

    """
    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)
        
        #
        self.visited_vis = None
        self.last_loc = None
        self.curr_loc = None
        
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.rgb_vis_frames = []
            self.goal_name = "No"
            
        # for diversity reward
        if args.use_diversity_reward:
            self.category_obj_id = {name:[] for name in target_coco_categories.keys()}
            
            for obj in self.habitat_env.sim.semantic_scene.objects[1:]:
                if obj.category.name() in target_coco_categories.keys():
                    self.category_obj_id[obj.category.name()].append(int(obj.id.replace('_', '')))
                    
            self.found_class = []
            self.found_id = []
            
        # for class entropy
        self.total_objects = 0
        self.history_counts = Counter()  # 记录所有历史帧中物体类别的频率
        self.prev_entropy = 0.0
        
        # for wypred
        device=torch.device("cuda:3")
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=device)
        
        
        
        cwp_fn = 'data_scene/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
            
        self.waypoint_predictor.to(device)
        self.waypoint_predictor.eval()
        model_config = {
        "depth_encoder": {
            "output_size": 128,          # 对应 forward 中的 depth 维度
            "ddppo_checkpoint": "data_scene/ddppo-models/gibson-2plus-resnet50.pth",
            "backbone": "resnet50",      # 或者 "resnet18"
        },
        # "rgb_encoder": {
        #     "output_size": 256,         # 对应 forward 中的 rgb 维度 (ResNet50 为 2048)
        #     "checkpoint": "path/to/rgb_ckpt.pth",
        #     "backbone": "resnet50",
        # },
        "spatial_output": False           # 必须为 True 才能输出 [C, H, W] 特征图
        }   
        dos = gym.spaces.Box(0., 1.0,
                        (args.env_frame_height,
                            args.env_frame_width, 1),
                        dtype='float32')
        
        self.depth_encoder = ResnetDepthEncoder(
            {'depth':dos},
            output_size=model_config['depth_encoder']['output_size'],
            checkpoint=model_config['depth_encoder']['ddppo_checkpoint'],
            backbone=model_config['depth_encoder']['backbone'],
            spatial_output=model_config['spatial_output'],
        ).to(device)
        self.rgb_encoder = CLIPEncoder(device)

    def reset(self):
        args = self.args
        
        obs, info = super().reset()
        obs, info = self._preprocess_obs(obs, info)

        # self.obs_shape = obs.shape

        # Episode initializations  TODO
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited_vis = np.zeros(map_shape)
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        
        # visualize
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend, mode=5)
        
        # for class entropy
        self.total_objects = 0
        self.history_counts = Counter()  # 记录所有历史帧中物体类别的频率
        self.prev_entropy = 0.0
        return obs, info
    
    def pred_wp_heatmap(self, observations,):
        batch_size = 1
        
        NUM_ANGLES = 120    # 360度划分为120个扇区，每个3度
        NUM_IMGS = 12      # 输入的12张环视图像
        NUM_CLASSES = 12   # 每个扇区预测12个距离等级 (0.25m - 3.0m)
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        # 2. 预处理：将输入图像从逆时针顺序转换为顺时针顺序（适应 Waypoint Predictor 训练分布）
        # H, W = observations['rgb'].shape[0], observations['rgb'].shape[1]
        rgb_batch = torch.zeros((NUM_IMGS, 224, 224, 3), dtype=torch.float32)
        depth_batch = torch.zeros((NUM_IMGS, 256, 256, 1), dtype=torch.float32)

        # 2. 遍历角度并进行坐标系转换与维度置换
        for i, angle in enumerate(angles):
            # 构建键名：0度直接用 'rgb'/'depth'，其余带后缀
            rgb_key = 'rgb' if angle == 0 else f'rgb_{angle}'
            depth_key = 'depth' if angle == 0 else f'depth_{angle}'
            
            # 计算目标索引：将逆时针索引转换为顺时针索引
            # i=0(0°) -> target=0
            # i=1(30°) -> target=11 (对应顺时针的 330°)
            # i=2(60°) -> target=10 (对应顺时针的 300°)
            target_idx = (NUM_IMGS - i) % NUM_IMGS
            
            # 获取数据并处理维度 (H, W, C) 
            rgb_img = torch.from_numpy(observations[rgb_key]).float().permute(2, 0, 1)
            depth_img = torch.from_numpy(observations[depth_key]).float().permute(2, 0, 1)
            # RGB resize 为 224*224
            rgb_tensor = F.interpolate(
                rgb_img.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Depth resize 为 256*256
            depth_tensor = F.interpolate(
                depth_img.unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
    
            # 填充到对应的 batch 位置
            rgb_batch[target_idx] = rgb_tensor.permute(1,2,0)
            depth_batch[target_idx] = depth_tensor.permute(1,2,0)

        obs_view12 = {'depth': depth_batch.to("cuda:3"), 'rgb': rgb_batch.to("cuda:3")}

        # 3. 特征提取
        # depth_embedding: [bs*12, 128, 4, 4], rgb_embedding: [bs*12, 2048, 7, 7]/[bs*12, 512]
        depth_embedding = self.depth_encoder(obs_view12)
        rgb_embedding = self.rgb_encoder(obs_view12)

        # 4. 热图预测 (模型输出 logits)
        # output shape: [batch_size, NUM_ANGLES * NUM_CLASSES] -> [bs, 1440]
        waypoint_heatmap_logits = self.waypoint_predictor(rgb_embedding, depth_embedding)

        # 5. 将 Logits 转换为概率分布 (Softmax)
        # 转换形状为 [Batch, 角度, 距离]
        # from heatmap to points
        batch_prob_map = torch.softmax(
            waypoint_heatmap_logits.reshape(batch_size, NUM_ANGLES * NUM_CLASSES), 
            dim=1
        ).reshape(batch_size, NUM_ANGLES, NUM_CLASSES)

        batch_x_norm_wrap = torch.cat((
            batch_prob_map[:,-1:,:], 
            batch_prob_map, 
            batch_prob_map[:,:1,:]), 
            dim=1)
        batch_output_map = nms(
            batch_x_norm_wrap.unsqueeze(1), 
            max_predictions=5,
            sigma=(7.0,5.0))

        # predicted waypoints before sampling
        batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

        # 6. (可选) 如果你的后续逻辑需要逆时针坐标系，在此处进行 Flip
        # 注意：原始代码在处理特征时进行了 flip，但在处理 heatmap 概率时通常保持顺时针，
        # 只有在最后计算 cand_angles 时才转回逆时针。
        
        return batch_output_map # Shape: [B, 120, 12]


    def step_and_preprocess(self, action, inputs):
        """Function responsible for taking the action and
        preprocessing observations

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W) ? 
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep
        """
        for f, inputs_f in enumerate(inputs):
            if f > len(self.rgb_vis_frames) - 1:
                continue
            # visualize 
            self.last_loc = self.curr_loc
            # Get Map prediction
            map_pred = np.rint(inputs_f['map_pred'])  # 四舍五入
        
            # Get pose prediction and global policy planning window
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                inputs_f['pose_pred']
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            
            # Get curr loc
            self.curr_loc = [start_x, start_y, start_o]
            r, c = start_y, start_x
            start = [int(r * 100.0 / self.args.map_resolution - gx1),
                    int(c * 100.0 / self.args.map_resolution - gy1)]
            start = pu.threshold_poses(start, map_pred.shape)
            
            if self.args.visualize or self.args.print_images:
                # Get last loc
                last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
                r, c = last_start_y, last_start_x
                last_start = [int(r * 100.0 / self.args.map_resolution - gx1),
                            int(c * 100.0 / self.args.map_resolution - gy1)]
                last_start = pu.threshold_poses(last_start, map_pred.shape)
                self.visited_vis[gx1:gx2, gy1:gy2] = \
                    vu.draw_line(last_start, start,
                                self.visited_vis[gx1:gx2, gy1:gy2])
                self._visualize(inputs_f, frame_id=f)

        # act and step
        # action = action + np.ones_like(action)   # output: 0-2, add to 1-3
        # action = {'action': action}
        obs_all, done_all, info = super().step(action)       # 4,256,256

        
        # preprocess obs
        obs_all, info = self._preprocess_obs(obs_all, info) # 改为对对个rgb序列的目标检测
        self.info = info
        self.timestep += 1  # 
        self.frameid = 0

        return obs_all, 0., done_all[-1], info
    
    def filter_instance(self, instance):
        '''
        实时推理并没有剪切模型，而是通过过滤结果类别
        '''
        
        if len(instance) == 0:
            return instance
        
        classes = instance.pred_classes
        new_instance = Instances(
                            pred_boxes=Boxes(torch.Tensor()),
                            image_size=instance.image_size,
                            pred_classes=torch.Tensor(),
                            pred_masks=torch.Tensor(),
                            scores=torch.Tensor(),
                        )
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx in list(target_coco_categories_mapping.keys()):
                new_instance = new_instance.cat([instance[i]])
            else:
                continue
        return new_instance
    
    def get_instance_id(self, semantic, category):
        '''
        返回gt的，当前frame的指定类别的所有物体ids
        '''
        object_ids = []
        for id in np.unique(semantic):
            if id > 0:
                if id in self.category_obj_id[category]:
                    object_ids.append(id)
        
        return len(object_ids) > 0, object_ids   
                     
    def _preprocess_obs(self, obs_seq, info, use_seg=True):
        args = self.args
        
        state_all = []
        self.rgb_vis_frames = []
        # for sequence reward
        obs_rgbs = []
        obs_detections = []
        
        for f, obs in enumerate(obs_seq):
            rgb = obs['rgb'].astype(np.uint8)
            depth = obs['depth']
            obs = np.concatenate((rgb, depth), axis=2)

            # obs = obs.transpose(1, 2, 0)
            
            rgb_ = obs[:, :, :3]     # 256,256,3
            depth_ = obs[:, :, 3:4]
        
        
            if args.det_frame_height != args.env_frame_height:
                rgb = cv2.resize(rgb_, (args.det_frame_height, args.det_frame_width))   #, rgb_.shape[-1]))
                depth = cv2.resize(depth_, (args.det_frame_height, args.det_frame_width))[..., None] #, 1))
            else:
                rgb = rgb_
                depth = depth_
            del rgb_
            del depth_

            return_score, return_instance = False, True     # return_score: use pred score as reward; 
            assert not (return_score and return_instance), \
                "Cannot return both score and instance at the same time."
            sem_seg_pred, obj = self._get_sem_pred(
                rgb.astype(np.uint8), use_seg=use_seg, return_score=return_score, return_instance=return_instance)
            # for sequence reward
            target_obj = self.filter_instance(obj)
            if f > self.info['no_straight_num'] - 1:
                obs_rgbs.append(obs_seq[f]['rgb'])
                obs_detections.append(target_obj)
            obs_detections.append(target_obj)
            depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

            ds = args.det_frame_width // args.frame_width  # Downscaling factor
            if ds != 1:
                rgb = np.asarray(self.res(rgb.astype(np.uint8)))
                depth = depth[ds // 2::ds, ds // 2::ds]
                sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

            depth = np.expand_dims(depth, axis=2)

            if return_instance:
                save_pred_ins = True if self.args.save_samples else False
                if save_pred_ins and f > self.info['no_straight_num'] - 1:
                    self.save_data({'bbspred': {'instances': obj}}, f)
                # TODO sequence 的相关奖励计算            
        
            state = np.concatenate((rgb, depth, sem_seg_pred),
                                axis=2).transpose(2, 0, 1)
            state_all.append(state)
        
        # # for sequence reward
        groups = group_by_object_and_score(obs_detections)
        num_frames = len(obs_detections)
        sequence_reward = 0.
        for g in groups:
            if g.has_object():
                g_class = int(list(g.history.values())[0].pred_classes.cpu().numpy())
                g_coeff = 0.01 / (self.history_counts.get(g_class, 0.) + 1. )
                g_r = 0.
                if g.frames[0] > 0:
                    g_r += g_coeff
                if g.frames[-1] < num_frames - 1:
                    g_r += g_coeff
                sequence_reward += g_r
        info['sequence_reward'] = sequence_reward

        # groups = score_tracks(groups, obs_detections)
        # groups = aggre_score_in_obj_tracks(groups, mode="frame")
        # groups = get_all_sample_score(clip_model, preprocess, text, groups, frame_rgbs)
        for od in obs_detections:
            if len(od) == 0:
                continue
            for det in range(len(od)):
                pred_cls = int(od[det].pred_classes.cpu().numpy())
                # if not pred_cls in list(target_coco_categories_mapping.keys()):
                #     continue
                self.history_counts.update([pred_cls])
                self.total_objects += 1
        
        
        curr_entropy = self._calculate_entropy()
        cls_etp = curr_entropy - self.prev_entropy
        info['cls_etp'] = cls_etp * 0.5
        
        self.prev_entropy = curr_entropy
        return state_all, info
    
    def _calculate_entropy(self):
        """
        根据当前频率分布计算 Shannon 熵 H_t
        """
        if self.total_objects == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.history_counts.values():
            p_i = count / self.total_objects
            if p_i > 0:
                entropy -= p_i * math.log(p_i)
        return entropy
        
    def obj_exist(self, instances_1, instances_2):
        """
        Check if instances from instances_1 exist in instances_2 by comparing their bounding boxes.
        
        Args:
            instances_1: First set of instances (Instances object) to check.
            instances_2: Second set of instances (Instances object) to compare against.
        
        Returns:
            A boolean array indicating whether each instance in instances_1 exists in instances_2 (IoU > 0.5).
        """
        '''
        instances_1: Instances(n), n is the number of instances
        check if every instances_1 exists in instances_2
        return [bool, bool, ...] of length n
        '''
        
        width, height = instances_1.image_size
        pot_mp = np.zeros((height, width, 1))
        v = Visualizer(pot_mp)
        
        boxes_1 = instances_1.pred_boxes
        boxes_1 = v._convert_boxes(boxes_1)
                
        boxes_2 = instances_2.pred_boxes
        boxes_2 = v._convert_boxes(boxes_2)
        iou = box_iou_calc(boxes_1, boxes_2) # 1*num_maskbox
        is_exists = iou > 0.5
        
        return is_exists
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
    
    def _get_sem_pred(self, rgb, use_seg=True, return_score=False, return_instance=False):
        if use_seg:
            semantic_pred, rgb_vis, obj = self.sem_pred.get_prediction(rgb, 
                                                                            return_instance=return_instance)
            semantic_pred = semantic_pred.astype(np.float32)
            self.rgb_vis_frames.append(rgb_vis)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 6))
            self.rgb_vis = rgb[:, :, ::-1]
        if not (return_instance or return_score):
            return semantic_pred
        else:
            return semantic_pred, obj
    
    
    def _visualize(self, inputs, mode="full", frame_id=0):
        
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)       # TODO, episode_no
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        sem_map = inputs['sem_map_pred']        # local map
        sem_map_full = inputs['sem_map_pred_full']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5        # 语义id，从5开始
        sem_map_full += 5        # 语义id，从5开始
        
        # local map
        no_cat_mask = sem_map == 10     # =最后一个通道，代表没有object
        map_mask = np.rint(map_pred) == 1       # obstacle地图
        exp_mask = np.rint(exp_pred) == 1       # explore地图
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1      # agent在地图上的位置

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2     # 将explore区域赋值2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1     # obstacle区域赋值1
        
        sem_map[vis_mask] = 3       # 可视化区域赋值3
        
        # add goal
        
        
        
        # full map
        map_pred_full = inputs['map_pred_full']
        exp_pred_full = inputs['exp_pred_full']
        no_cat_mask_full = sem_map_full == 10     # 最后一个通道是什么
        map_mask_full = np.rint(map_pred_full) == 1
        exp_mask_full = np.rint(exp_pred_full) == 1
        vis_mask_full = self.visited_vis == 1

        sem_map_full[no_cat_mask_full] = 0
        m1_full = np.logical_and(no_cat_mask_full, exp_mask_full)
        sem_map_full[m1_full] = 2     # 将explore区域赋值2

        m2_full = np.logical_and(no_cat_mask_full, map_mask_full)
        sem_map_full[m2_full] = 1     # obstacle区域赋值1

        sem_map_full[vis_mask_full] = 3       # agent位置区域赋值3

        # orient map
        orient_full = np.zeros((495, 750), dtype=np.uint8)
        sub_size = 240
        gap = 15
        for i in range(6):
            obs_key = f'orient_full_map_{i}_obsta'
            exp_key = f'orient_full_map_{i}_exp'
            
            if obs_key in inputs and exp_key in inputs:
                # 1. 提取并合成语义数据
                o_obs = inputs[obs_key]
                o_exp = inputs[exp_key]
                
                # 创建单通道数据：0背景，1障碍(obsta)，2探索(exp)
                o_sem = np.zeros_like(o_obs, dtype=np.uint8)
                o_sem[np.rint(o_exp) == 1] = 2
                o_sem[np.rint(o_obs) == 1] = 1
                
                # 2. 转换为彩色图 (使用与主图相同的 color_palette)
                # o_vis_img = Image.new("P", (o_sem.shape[1], o_sem.shape[0]))
                # o_vis_img.putpalette(color_pal)
                # o_vis_img.putdata(o_sem.flatten().astype(np.uint8))
                # o_vis_img = o_vis_img.convert("RGB")
                
                # o_vis_img = np.flipud(o_vis_img)
                # o_vis_bgr = np.array(o_vis_img)[:, :, [2, 1, 0]]
                
                o_vis_res = cv2.resize(o_sem, (sub_size, sub_size), 
                                      interpolation=cv2.INTER_NEAREST)
                
                # 5. 计算在 orient_vis (495, 750) 上的位置
                # 两排三列布局
                row_idx = i // 3  # 0, 1
                col_idx = i % 3   # 0, 1, 2
                
                y_start = row_idx * (sub_size + gap)
                x_start = col_idx * (sub_size + gap)
                
                # 将小图贴到 orient_vis 画布上
                orient_full[y_start:y_start + sub_size, 
                           x_start:x_start + sub_size] = o_vis_res
        
        
        
        if 'frontier_goal' in inputs:
            if inputs['frontier_goal'] is not None:
                goal = inputs['frontier_goal']
                goal_r, goal_c = goal   # r,c
                goal_x = goal_r
                goal_y = goal_c
                
                st_goal = inputs['short_time_goal']
                st_goal_r, st_goal_c = st_goal
                st_goal_r, st_goal_c = int(st_goal_r), int(st_goal_c)
                st_goal_x = st_goal_r
                st_goal_y = st_goal_c

                size = self.visited_vis.shape[0]
                square_size = 20
                half_size = square_size // 2
                for i in range(goal_x - half_size, goal_x + half_size + 1):
                    for j in range(goal_y - half_size, goal_y + half_size + 1):
                        i = min(i, size-1)
                        j = min(j, size-1)
                        sem_map_full[i, j] = 12
                        
                square_size = 10
                half_size = square_size // 2
                for i in range(st_goal_x - half_size, st_goal_x + half_size + 1):
                    for j in range(st_goal_y - half_size, st_goal_y + half_size + 1):
                        i = min(i, size-1)
                        j = min(j, size-1)
                        sem_map_full[i, j] = 12
                        
        # pos
        # size = map_pred_full.shape[0]
        # square_size = 10
        # r, c = start_y, start_x     # 转化为格子坐标
        # start = [int(r * 100.0 / args.map_resolution),
        #         int(c * 100.0 / args.map_resolution)]
        # # start[1] = map_pred_full.shape[0] - start[1] 
        # start = pu.threshold_poses(start, map_pred_full.shape)
        # half_size = square_size // 2
        # for i in range(start[0] - half_size, start[0] + half_size + 1):
        #     for j in range(start[1] - half_size, start[1] + half_size + 1):
        #         i = min(i, size-1)
        #         j = min(j, size-1)
        #         sem_map_full[i, j] = 17
        # 绘制语义地图
        color_pal = [int(x * 255.) for x in color_palette]
        if mode == "local":
            sem_map_vis = Image.new("P", (sem_map.shape[1],
                                        sem_map.shape[0]))
            sem_map_vis.putpalette(color_pal)
            sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        elif mode == "full":        
            sem_map_vis = Image.new("P", (sem_map_full.shape[1],
                                        sem_map_full.shape[0]))
            sem_map_vis.putpalette(color_pal)
            sem_map_vis.putdata(sem_map_full.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
        
        rgb_vis = cv2.resize(self.rgb_vis_frames[frame_id], (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        
        self.vis_image[50:530, 15:495] = rgb_vis
        self.vis_image[50:530, 510:990] = sem_map_vis
        
        
        # for orient map
        orient_vis = Image.new("P", (orient_full.shape[1],
                                        orient_full.shape[0]))
        orient_vis.putpalette(color_pal)
        orient_vis.putdata(orient_full.flatten().astype(np.uint8))
        orient_vis = orient_vis.convert("RGB")
        orient_vis = np.flipud(orient_vis)
        
        # for i in range(6):
        #     row_idx = i // 3
        #     col_idx = i % 3
        #     y_start = row_idx * (sub_size + gap)
        #     x_start = col_idx * (sub_size + gap)
            
        #     # 文本内容和位置
        #     text = f"Orient {i}"
        #     # 这里的坐标 (x, y) 是文字左下角
        #     text_pos = (x_start + 100, y_start + 25) 
            
        #     # 绘制黑边阴影（可选，增加可读性）
        #     # cv2.putText(orient_vis, text, (text_pos[0]+1, text_pos[1]+1),
        #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        #     # 绘制白色主文字
        #     cv2.putText(orient_vis, text, text_pos,
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        
        orient_vis = orient_vis[:, :, [2, 1, 0]]
        self.vis_image[50:545, 1005:1755] = orient_vis
        
        # 绘制agent位置
        if mode == "local":
            pos = (
                (start_x * 100. / args.map_resolution - gy1)        # start_x是full pose, 所以减去local bdry得到local pose
                * 480 / map_pred.shape[0],
                (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
                * 480 / map_pred.shape[1],
                np.deg2rad(-start_o)
            )
        elif mode == "full":
            pos = (
                (start_x * 100. / args.map_resolution)
                * 480 / map_pred_full.shape[0],
                (map_pred_full.shape[1] - start_y * 100. / args.map_resolution)
                * 480 / map_pred_full.shape[1],
                np.deg2rad(-start_o)
            )
            
            
            
        origin = (510, 50)  
        agent_arrow = vu.get_contour_points(pos, origin)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)
            pass

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep, frame_id)
            cv2.imwrite(fn, self.vis_image)
