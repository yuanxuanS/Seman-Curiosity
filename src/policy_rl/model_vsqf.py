import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

import numpy as np

from .utils.distributions import Categorical, DiagGaussian
from .utils.model import get_grid, ChannelPool, Flatten, NNBase
from .envs.utils import depth_utils as du
import math
import itertools

class VSQF_Mapping(nn.Module):

    """
    VSQF_Mapping
    """

    def __init__(self, args):
        super(VSQF_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        
        ### 生成vsqf区域内所有坐标
        diameter = 11 * 100   # 场的实际直径(cm)
        self.center = (diameter / 2, diameter / 2)
        vr = int(diameter // self.resolution)       # vsqf vision range

        # 直接生成坐标轴数组， x向右，y向下
        x = np.arange(0, diameter, 10)
        y = np.arange(0, diameter, 10)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        self.coords = np.column_stack((xx.ravel(), yy.ravel())) # coords: num * 2, 实际坐标值（单位m）
        self.distance_center = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]    # distance_center: [n_distance_bin], 距离分区的中心
        self.distance_center = [i*100 for i in self.distance_center]
        
        self.init_grid = torch.zeros(
            args.num_processes, 1, vr, vr,
        ).float().to(self.device)       # cls+1, h, w, z
        
        self.coord_range = vr
        self.fov = args.hfov
        
        # 
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)
        
        self.dist_idx, self.angle_idx, self.valid_mask = self.reset()
        
        
        
        
    def reset(self):
        coords = self.coords.copy()
        
        # 根据半径填充不同区域
        dy = coords[:, 0] - self.center[0]       # x为行索引，y方向
        dx = coords[:, 1] - self.center[1]
        distance_array = np.sqrt(dx**2 + dy**2)
        distance_array = torch.from_numpy(distance_array)
        # print(distance_array.shape)

        # 根据距离矩阵计算每个坐标所属的距离区间
        dis_list = [d-25 for d in self.distance_center]  # 0.25 - 5.25
        dis_list.append(dis_list[-1]+25)      # 0.25 = 5.25 , 5.75
        dis_bins = torch.tensor([0] + dis_list + [float('inf')], dtype=torch.float32)
        dist_idx = torch.bucketize(distance_array, dis_bins, right=True) - 1
        # 过滤无效索引 (距离超出 dis3 或角度无效)
        valid_mask = (dist_idx > 0) & (dist_idx < len(dis_list))  # 有效区域掩码
        dist_idx = dist_idx[valid_mask] - 1
        
        # 根据角度计算每个坐标所属角度区间
        angle_rad = torch.atan2(torch.from_numpy(-dx[valid_mask]), torch.from_numpy(-dy[valid_mask]))    # 极坐标方向为x正，逆时针；弧度 [-π, π]
        # angle_rad = angle_rad[valid_mask]
        angle_deg = (angle_rad * 180 / math.pi)
        angle_deg = angle_deg % 360  # 转换为 [0°, 360°)
        # 计算角度索引 (0-35 对应 0°-350°)
        # angle_idx = (angle_deg // 10).long()  # 每10°一个索引 [height, width]
        shifted_angles = (angle_deg + 5) % 360
        boundaries = torch.linspace(0, 360, 37)
        indices = torch.bucketize(shifted_angles, boundaries, right=False)
        angle_idx = (indices - 1) % 36    
        return dist_idx, angle_idx, valid_mask
        
    def splat_value_in_field(self, vsqf):
        '''
        根据价值矩阵，将每个区域的点赋值；
            
            vsqf: 价值矩阵，num_scenes* n_distance_bin * num_angle_bin
            竖直向上为azimuth正向
        返回有效圆内的坐标点 coords，和点对应的值values
            values: num_scenes * num_pts
            coords: num_scenes * num_pts* 2
        '''
        num_scenes = vsqf.shape[0]
        dist_idx = self.dist_idx.unsqueeze(0).repeat(num_scenes, 1)
        dist_idx = dist_idx.to(vsqf.device)
        angle_idx = self.angle_idx.unsqueeze(0).repeat(num_scenes, 1)
        angle_idx = angle_idx.to(vsqf.device)

        depth_indices = torch.arange(num_scenes)[..., None].repeat(1, dist_idx.shape[-1])
        depth_indices = depth_indices.to(vsqf.device)
        
        values = vsqf[depth_indices, dist_idx, angle_idx]        # dim=1
        coords = self.coords[self.valid_mask]
        
        return values, coords[None, ...].repeat(num_scenes, 0)

    def splat_field_in_map(self, init_grid, feat, coords):
            
        '''
        和envs/utils/depth_utils.py 的splat_feat_nd相同，区别是对scatter_add可修改
        将coords对应的特征feat， 按照坐标值赋值到地图init_grid中
        对grid和coord不同维度的,grid的dim和coord一致即可, 
        如:
            grid: [b, F, w, h]
            coords: [b, nDim=2, nPt]
        '''

        wts_dim = []        # 按地图维度顺序的对应权重
        pos_dim = []
        grid_dims = init_grid.shape[2:]
        
        B = init_grid.shape[0]
        F = init_grid.shape[1]
        n_dims = len(grid_dims)
        
        grid_flat = init_grid.view(B, F, -1)
        
        
        for d in range(n_dims):
            pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
            
            pos_d = []
            wts_d = []
            # 每个坐标点为浮点数，根据和网格整索引的差距作为权重，加权该坐标的特征值到网格
            for ix in [0, 1]:
                pos_ix = torch.floor(pos) + ix      # b * 1 * nPt
                safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d]) 
                safe_ix = safe_ix.type(pos.dtype)
                
                wts_ix = 1 - torch.abs(pos - pos_ix)        # b, nPt, 权重为距离整数索引的浮点值

                # 留下有效索引
                wts_ix = wts_ix * safe_ix
                pos_ix = pos_ix * safe_ix
                
                pos_d.append(pos_ix)
                wts_d.append(wts_ix)

            pos_dim.append(pos_d)   
            wts_dim.append(wts_d)
            
        l_ix = [[0, 1] for d in range(n_dims)]
        
        for ix_d in itertools.product(*l_ix): 
            wts = torch.ones_like(wts_dim[0][0], dtype=torch.float32)        # b*1*nPt
            index = torch.zeros_like(wts_dim[0][0])
            
            # 计算展为一维后的索引
            for d in range(n_dims):
                index = index * grid_dims[d] + pos_dim[d][ix_d[d]]
                wts = wts * wts_dim[d][ix_d[d]]
            
            index = index.long()
            wts = wts.type(torch.float32)
            grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)     # dim为展开的维度 ;TODO: 不使用add而是该位置的值
            grid_flat = torch.round(grid_flat)
            
        grid = grid_flat.view(init_grid.shape)
        return grid

    def forward(self, vsqf, azimuth, depth_obj, pose_obs, maps_last, poses_last):
        '''
        azimuth: num_scenes*1
        '''
        B = vsqf.shape[0]
        
        values, coords = self.splat_value_in_field(vsqf)
        feat = values[:, None, ...]     # B*nF*nPt

        XY = coords / self.resolution     # 像素地图的坐标， B*2*nPt
        XY = (XY - self.coord_range // 2) / self.coord_range * 2      # 除以坐标最大范围，所有坐标映射到 [-1, 1]
        # print(XY.shape)
        XY= XY.transpose(0, 2, 1)      # b X n_dim=2 X length
        XY = torch.from_numpy(XY)
        
        quality_field = self.splat_field_in_map(
            self.init_grid.to(feat.device) * 0., feat, XY.to(feat.device)
        )       # B*1*vr*vr
        
        # 转到azimuth角度
        obj_pose = torch.zeros(B, 3)
        obj_pose[:, 2] = -azimuth
        obj_pose = obj_pose.to(self.device)
        rot_mat, trans_mat = get_grid(obj_pose, quality_field.size(),
                                        self.device)
        rotated_qf = F.grid_sample(quality_field.to(self.device), rot_mat, align_corners=True)

        # obj和agent的相对距离
        
        point_cloud_t = du.get_point_cloud_from_z_t(
            torch.from_numpy(depth_obj).to(self.device), 
            self.camera_matrix, 
            self.device, 
            scale=1)
        
        dx_obj = point_cloud_t[..., 0].mean()        
        dy_obj = (point_cloud_t[..., 1].mean() * 4.5 + 0.5) * (point_cloud_t[..., 1].mean() > 0)       # 深度方向
        # print(f"obj x {dx_obj}, y {dy_obj}")
        
        pose_pred = poses_last
        ### vsqf到 agent view的 全局地图上
        agent_view = torch.zeros((B, 1,
                            int(self.map_size_cm // self.resolution),
                            int(self.map_size_cm // self.resolution)
                            )).to(self.device)

        x1 =int(self.map_size_cm // (self.resolution * 2) - self.coord_range // 2 - dx_obj // self.resolution)
        x2 = x1 + self.coord_range
        y1 = int(self.map_size_cm // (self.resolution * 2) + dy_obj // self.resolution) # - vision_range // 2)     # field以地图中心为原点 ？
        y2 = min(y1 + self.coord_range, int(self.map_size_cm // self.resolution))
        y_range = y2 - y1
        rotated_qf = rotated_qf[:, :, :int(y_range), :]
        agent_view[:, :, y1:y2, x1:x2] = rotated_qf
        
        # 转换到world map
        corrected_pose = pose_obs       # 相对t-1的pose1的改变, dx, dy, do (在psoe1坐标系)
        
        def get_new_pose_batch(pose, rel_pose_change):
            '''
                返回世界坐标系下新的pose
            '''
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)       # 局部位移 (dx, dy)通过旋转矩阵转换到世界坐标系， dy_world
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232     # do_world

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0      # -180, 180范围

            return pose
        
        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        # 坐标方向 x向右y向下；移动到地图中心，并归一化到 [-1, 1], 坐标系正向反转（affine函数的坐标为 x向左y向上）
        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])       # 向上，顺时针角度增加, 
        
        # 将当前地图进行平移+旋转，和上一时刻地图进行融合
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # update map with last map
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)
        
        return map_pred, pose_pred, current_poses


if __name__ == "__main__":
    import gym

    observation_space = gym.spaces.Box(0, 255,
                                                (3, 128,
                                                 128),
                                                dtype='uint8')
    action_space = gym.spaces.Discrete(3)
    print(f"obs shape:{observation_space.shape}")
    
    device = "cuda:1"
    policy = RL_Policy(observation_space.shape,
                action_space, model_type=1,
                base_kwargs={'recurrent': True,
                                      'hidden_size': 512,
                                    #   'num_sem_categories': args.num_sem_categories
                                      })

    bs = 10
    rnn_hxs = torch.rand(bs, 512)
    l_masks = torch.ones(bs)
    extras = None
    input = torch.rand(bs, 3, 128, 128)
    # value, action_feature, rnn_hxs = policy(input, rnn_hxs, l_masks, extras)
    # print(f"shape: \nvalue:{value.shape}\naction feature:{action_feature.shape}\n")

    value, action, action_prob, rnn_hxs = policy.act(input, rnn_hxs, l_masks, extras)
    print(f"value: {value}")
    print(f"action: {action}")
    print(f"action_prob: {action_prob}")
    print(f"rnn_hxs: {rnn_hxs.shape}")
    