import torch
import math
import numpy as np
import itertools
from src.finetune.dataset_utils import get_loader, SampleLoader
from torch.nn import functional as F
from src.policy_rl.utils.model import get_grid
from PIL import Image
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug
from third_parties.Orient_Anything.utils import background_preprocess
from transformers import AutoImageProcessor
import cv2

def polar_to_cartesian_map(values_2d, dis_list, map_size, origin=(0, 0)):
    """
    将极坐标值映射到笛卡尔坐标系地图
    :param values_2d: torch.Tensor, 形状为 [3, 36] 的二维数组
    :param dis_list: list, 距离列表 [dis1, dis2, dis3]
    :param map_size: tuple, 地图尺寸 (width, height); 半径5.5m, 11m宽
    :param origin: tuple, 原点坐标 (cx, cy)
    :return: torch.Tensor, 赋值后的地图
    """
    width, height = map_size
    cx, cy = origin  # 原点坐标: 物体原点
    
    # 1. 生成网格坐标 (向量化)
    i = torch.arange(height, dtype=torch.float32)  # 行索引
    j = torch.arange(width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(i, j, indexing='ij')  # 形状: [height, width]
    
    # 2. 计算相对原点的坐标
    dx = grid_x - cx  # x方向偏移
    dy = grid_y - cy  # y方向偏移
    dist = torch.sqrt(dx**2 + dy**2)  # 距离矩阵， [height, width]： 每个位置距离物体中心的距离
    angle_rad = torch.atan2(dy, dx)    # 弧度 [-π, π]
    angle_deg = (angle_rad * 180 / math.pi) % 360  # 转换为 [0°, 360°)
    
    # 3. 计算角度索引 (0-35 对应 0°-350°)
    angle_idx = (angle_deg // 10).long()  # 每10°一个索引 [height, width]
    
    # 4. 计算距离索引 (0/1/2 对应 dis1/dis2/dis3)
    # 生成距离分界点: [0, dis1, dis2, dis3, ∞]
    dis_bins = torch.tensor([0] + dis_list + [float('inf')], dtype=torch.float32)
    dist_idx = torch.bucketize(dist, dis_bins, right=True) - 1  # dist中每个元素所属区间的索引； 映射到 0/1/2 [height, width]
    
    # 5. 过滤无效索引 (距离超出 dis3 或角度无效)
    valid_mask = (dist_idx >= 0) & (dist_idx < len(dis_list))  # 有效区域掩码
    
    # 6. 通过索引赋值
    map_result = torch.zeros((height, width), dtype=values_2d.dtype)
    # 使用 valid_mask 过滤后赋值  ?
    rows, cols = torch.where(valid_mask)
    valid_dist = dist_idx[rows, cols]
    valid_angle = angle_idx[rows, cols]
    map_result[valid_mask] = values_2d[dist_idx[valid_mask], angle_idx[valid_mask]].reshape(height, width)
    
    return map_result

def splat_field_in_map(init_grid, feat, coords):
        
    '''
    将coords对应的特征feat， 按照坐标值赋值到地图init_grid中
    对grid和coord不同维度的,grid的dim和coord一致即可, 
    如:
        grid: [b, F, w, h]
        coords: [b, nDim=2, nPt]
    '''

    wts_dim = []        # 按地图维度顺序的对应权重
    pos_dim = []
    grid_dims = init_grid.shape[2:]
    
    F = init_grid.shape[1]
    n_dims = len(grid_dims)
    
    grid_flat = init_grid.view(b, F, -1)
    
    
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

def splat_value_in_field(center, coords, distance_center, value_array):
    '''
    根据价值矩阵，将每个区域的点赋值；
        coords: num * 2, 实际坐标值（单位m）
        distance_center: [n_distance_bin], 距离分区的中心
        value_array: 距离矩阵，n_distance_bin * num_angle_bin
        竖直向上为azimuth正向
    返回有效圆内的坐标点 coords，和点对应的值values
    '''
    
    # 根据半径填充不同区域
    dy = coords[:, 0] - center[0]       # x为行索引，y方向
    dx = coords[:, 1] - center[1]
    distance_array = torch.from_numpy(np.sqrt(dx**2 + dy**2))
    print(distance_array.shape)

    # 根据距离矩阵计算每个坐标所属的距离区间
    dis_list = [d-0.25 for d in distance_center]  # 0.25 - 5.25
    dis_list.append(dis_list[-1]+0.25)      # 0.25 = 5.25 , 5.75
    dis_bins = torch.tensor([0] + dis_list + [float('inf')], dtype=torch.float32)
    dist_idx = torch.bucketize(distance_array, dis_bins, right=True) - 1
    # 过滤无效索引 (距离超出 dis3 或角度无效)
    valid_mask = (dist_idx > 0) & (dist_idx < len(dis_list))  # 有效区域掩码
    dist_idx = dist_idx[valid_mask] - 1
    
    # 根据角度计算每个坐标所属角度区间
    angle_rad = torch.atan2(torch.from_numpy(-dx[valid_mask]), torch.from_numpy(-dy[valid_mask]))    # 极坐标方向为x正，逆时针；弧度 [-π, π]
    # angle_rad = angle_rad[valid_mask]
    angle_deg = (angle_rad * 180 / math.pi)
    print(f"max {angle_deg.max()}, min {angle_deg.min()}")
    angle_deg = angle_deg % 360  # 转换为 [0°, 360°)
    # 计算角度索引 (0-35 对应 0°-350°)
    # angle_idx = (angle_deg // 10).long()  # 每10°一个索引 [height, width]
    shifted_angles = (angle_deg + 5) % 360
    boundaries = torch.linspace(0, 360, 37)
    indices = torch.bucketize(shifted_angles, boundaries, right=False)
    angle_idx = (indices - 1) % 36    
    
    values = value_array[dist_idx, angle_idx]        # dim=1
    coords = coords[valid_mask]
    
    return values, coords
    
if __name__ == "__main__":
    
    ### 生成vsqf区域内所有坐标
    diameter = 11   # 场的实际直径(m)
    center = (diameter / 2, diameter / 2)

    # 直接生成坐标轴数组， x向右，y向下
    x = np.arange(0, diameter, 0.1)
    y = np.arange(0, diameter, 0.1)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    
    distance_center = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    
    ### 加载图像
    dataset_path = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/obj_samples/data/Collierville"
    sampler = SampleLoader(dataset_path)
    env, episode, step = 1, 0, 159      # 24
    obj_id = 39
    sample_data = sampler.get_sample_multimodality(env, episode, step, ["bbsgt", "rgb", "depth", "semantic"])
    rgb = sample_data['rgb'].data
    instance = sample_data['bbsgt'].data
    depth = sample_data['depth'].data
    semantic = sample_data['semantic'].data
    # semantic == objid +depth得到agent坐标系下物体中心: 点云，计算点云中心    

    mask = semantic == obj_id
    rgb_obj = rgb * mask[:, :, None]
    depth_obj = depth * mask[:, :, None]
    
    rgb_img = Image.fromarray(rgb_obj)  # .convert('RGB')
    
    ## VSQF预测
    load_model = True
    model_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/08-29_17-25-03_/best_unseen_model_e1.pth"
    magnify = 3.
    from vqf import VQFModel
    from train_vqf import visualize
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = VQFModel(device)
    if load_model:
        model.load_state_dict(torch.load(model_pth))
    
    with torch.no_grad():
        x = rgb_img
        x = model.preprocess(x).to(device)[None, ...]
        vsqf = model(x)
    visualize(vsqf.squeeze(0), "images", "./", "vsqf")
    ### 得到vqsf每个坐标对应的值
    vsqf = (vsqf.squeeze(0) - vsqf.min() + 0.1) * magnify / (vsqf.max() - vsqf.min())
    values, coords = splat_value_in_field(center, coords, distance_center, vsqf)
    
    ### 分散vsqf到2D地图上
    XY_resolution = 0.05     # 单位,m
    w, h = int(11 / XY_resolution), int(11 / XY_resolution)       # field地图大小
    b = 1
    init_grid = torch.zeros(b, 1, w, h)
    feat = values[None, None, ...].cpu()
    
    XY = coords / XY_resolution     # 地图分辨率的坐标
    coord_range = diameter / XY_resolution
    XY = (XY - coord_range // 2) / coord_range * 2      # 除以坐标最大范围，所有坐标映射到 [-1, 1]
    print(XY.shape)
    XY= XY[None, ...].transpose(0, 2, 1)      # b X n_dim=2 X length
    XY = torch.from_numpy(XY)
    
    # 在地图上x向右，y向下，agent朝向y正；
    quality_field = splat_field_in_map(init_grid, feat, XY) # 将坐标XY对应的值feat赋值到init_grid中
    
        
        
    ### 得到agent为中心的vsqf 2D map
    from src.policy_rl.envs.utils import depth_utils as du

    ## 根据点云计算obj相对agent的位置、朝向
    screen_w, screen_h = 128, 128  #?
    fov = 79
    device = "cpu"
    camera_matrix = du.get_camera_matrix(
            screen_w, screen_h, fov)
    
    ## TODO 加载两个时刻的信息，叠加 1. gt_orient + gt的field 2. OriAny orient + gt的field
    #   两个时刻分别得到affine后的field， 叠加
    ## 如何从field中得到reward？

    point_cloud_t = du.get_point_cloud_from_z_t(
            torch.from_numpy(depth_obj), camera_matrix, device, scale=1)
    
    # agent_view_t = du.transform_camera_view_t(
    #     point_cloud_t, self.agent_height, 0, self.device)
    
    dx_obj = point_cloud_t[..., 0].mean()        
    dy_obj = point_cloud_t[..., 1].mean() * 4.5 + 0.5       # 深度方向
    
    
    ### vsqf到 agent view的 全局地图上
    map_size_m = 24
    agent_view = torch.zeros((1, 1,
                        int(map_size_m // XY_resolution),
                        int(map_size_m // XY_resolution)
                        )).to(device)
    
    vision_range = w # 和field最小外接矩形的边长一致
    x1 =int(map_size_m // (XY_resolution * 2) - vision_range // 2 )
    x2 = x1 + vision_range
    y1 = int(map_size_m // (XY_resolution * 2)) # - vision_range // 2)     # field以地图中心为原点 ？
    y2 = y1 + vision_range
    
    agent_view[:, :, y1:y2, x1:x2] = quality_field
    
    # 预测agent在obj的相对朝向 
    augment = False
    
    from test_gt_orient import OriAny_pred
    dino, val_preprocess = OriAny_pred(device)
    # rgb_img = Image.fromarray(rgb_obj).convert('RGB')
    if augment:
        rm_bkg_img = background_preprocess(rgb_img, True)
        angles = get_3angle_infer_aug(rgb_img, rm_bkg_img, dino, val_preprocess, device)
    else:
        angles = get_3angle(rgb_img, dino, val_preprocess, device)

    azimuth     = float(angles[0])
    polar       = float(angles[1])
    rotation    = float(angles[2])
    confidence  = float(angles[3])
                
    print(f"dx {dx_obj}, dy {dy_obj}, azimuth {azimuth}")

    x_obj = -dx_obj    #map_size_m // 2 - dx_obj    # obj在agent为中心的地图上的位置, obj相对agent的x正向，此时为地图的-x
    y_obj = dy_obj    # map_size_m // 2 + dy_obj
    
    ## 得到agent为中心的场
    # 归一化到 [-1, 1]；（以及取负都是affine函数需要）
    x_obj = - (x_obj / XY_resolution - map_size_m // (XY_resolution * 2)) / (map_size_m // (XY_resolution * 2))
    y_obj = - (y_obj / XY_resolution - map_size_m // (XY_resolution * 2)) / (map_size_m // (XY_resolution * 2))
    obj_pose = torch.tensor([x_obj, y_obj, -azimuth]).unsqueeze(0)
    
    rot_mat, trans_mat = get_grid(obj_pose, agent_view.size(),
                                      device)
    rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
    translated_agent_view = F.grid_sample(rotated, trans_mat, align_corners=True)
    print(f"max value in vsqf map {translated_agent_view.max()}, min {translated_agent_view.min()}")
    # 经过agent pose转换到全局坐标中
    
    # visualize
    import matplotlib.pyplot as plt
    # 假设 tensor 为输入数据，尺寸 [1, 1, w, h]
    data_orig = agent_view.squeeze(0).squeeze(0).detach().cpu().numpy()  # 压缩为 [w, h]
    data = translated_agent_view.squeeze(0).squeeze(0).detach().cpu().numpy()  # 压缩为 [w, h]

    
    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].imshow(data_orig, cmap='hot')
    ax[0].set_title("Original")
    ax[1].imshow(translated_agent_view.squeeze().detach(), cmap='hot')
    ax[1].set_title("Rotated 30° + Right Shift")
    # plt.show()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.savefig('test_valuemap_affine9.png')