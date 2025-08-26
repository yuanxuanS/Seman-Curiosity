import torch
import math
import numpy as np
import itertools

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

def splat_value_in_map(init_grid, feat, coords):
        
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

def splat_value_in_correspond_region(center, coords, distance_center, value_array):
    '''
    coords: num * 2
    value_array: n_distance_bin * num_angle_bin
    竖直向下为azimuth正向
    '''
    
    # 根据半径填充不同区域
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
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
    angle_rad = torch.atan2(torch.from_numpy(dx), torch.from_numpy(-dy))    # 弧度 [-π, π]
    angle_rad = angle_rad[valid_mask]
    angle_deg = (angle_rad * 180 / math.pi) % 360  # 转换为 [0°, 360°)
    angle_rad = angle_rad % 360
    # 计算角度索引 (0-35 对应 0°-350°)
    angle_idx = (angle_deg // 10).long()  # 每10°一个索引 [height, width]
    
    
    # value_list = [1e4] + value_list
    # value_list.append(1e4)
    # value_list = torch.tensor(value_list)
    # value_array = torch.ones(len(distance_center), 36)
    # value_array[1, :] *= 4
    # value_array[5, :] *= 2
    # value_array[7, :] *= 6
    # value_array[:, 4] *=2
    
    values = value_array[dist_idx, angle_idx]        # dim=1
    coords = coords[valid_mask]
    
    return values, coords
    
if __name__ == "__main__":
    # 示例输入参数
    values_2d = torch.randn(3, 36)  # 3×36 的二维数组
    dis_list = [10.0, 20.0, 30.0]   # 距离层级
    map_size = (200, 200)            # 地图宽高
    origin = (100, 100)              # 原点坐标（地图中心）
    
    # 创建实际坐标，直径11m, 间隔小于0.5m
    # coords = []
    center = (5.5, 5.5)

    # 直接生成坐标轴数组， x向右，y向下
    x = np.arange(0, 11, 0.1)
    y = np.arange(0, 11, 0.1)

    # 使用meshgrid生成网格坐标，再组合成点集
    xx, yy = np.meshgrid(x, y, indexing='xy')
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    
    distance_center = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    value_list = [i+1 for i in range(len(distance_center))]
    values, coords = splat_value_in_correspond_region(center, coords, distance_center, value_list)
    
    w, h = 40, 40
    b = 1
    init_grid = torch.zeros(b, 1, w, h)
    feat = values[None, None, ...]
    XY_resolution = 0.3

    
    print(coords.shape)
    
    print(coords.max(), coords.min())
    XY = coords / XY_resolution     # 地图分辨率的坐标
    print(XY.max(), XY.min())
    coord_range = 11 / XY_resolution
    XY = (XY - coord_range // 2) / coord_range * 2      # 除以坐标最大范围，所有坐标映射到 [-1, 1]
    print(XY.max(), XY.min(), XY.shape)
    print(type(XY))
    XY= XY[None, ...].transpose(0, 2, 1)      # b X n_dim=2 X length
    print(XY.shape)
    XY = torch.from_numpy(XY)
    
    # 
    grid = splat_value_in_map(init_grid, feat, XY)
    print(grid.shape)
    
    # visualize
    import matplotlib.pyplot as plt
    # 假设 tensor 为输入数据，尺寸 [1, 1, w, h]
    data = grid.squeeze(0).squeeze(0).detach().cpu().numpy()  # 压缩为 [w, h]

    # 方法1：Matplotlib imshow
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', vmin=0, vmax=100)  # vmin/vmax 限定值范围
    plt.colorbar(label='Value')  # 添加颜色条
    plt.title("Tensor俯视图")
    plt.axis('off')  # 隐藏坐标轴
    # plt.show()
    plt.savefig("./t_valuemap3.png")