from skimage.draw import line
from typing import Tuple, Set, List
from scipy.interpolate import griddata
from skimage.morphology import binary_erosion
import numpy as np
import math
import cv2
import torch
import os
from multiprocessing import Pool

def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output
    Inputs:
        mu: tensor (N, 2); 0dim = col, 2dim=row
        
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    # Generate bivariate Gaussians centered at position mu
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma[0])**2 + (y_diff/sigma[1])**2 ))
    else:
        output = torch.logical_and(
            torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]
        ).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=(1.0,1.0), gaussian=False):
    ''' Input (batch_size, 1, height, width) '''

    shape = pred.shape

    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)

    for i in range(max_predictions):
        # Find and save max over the entire map
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        flat_output[indices,ix] = flat_pred[indices,ix]

        # Suppression
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float()

        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        supp_pred *= (1-g)

    # output[output < 0] = 0
    return flat_output.reshape((shape[0],shape[1], shape[2]))

def check_ray_vectorized(ray_path_rr, ray_path_cc, H, W, exp_map, wall_mask, visibility_mask):
    
    if len(ray_path_cc) == 1:
        visibility_mask[ray_path_rr, ray_path_cc] == True
        return visibility_mask
    
    # --- 1. 裁剪路径到地图边界内 (可选但推荐) ---
    # 通常在高效的 line 实现中，路径已经处理过边界
    
    # --- 2. 一次性查询所有像素的状态 ---
    
    # 路径上的所有点是否为墙壁？ (True=阻挡)
    # 注意：需排除路径上的第一个点 (T_j)，因为它是起点
    is_wall = wall_mask[ray_path_rr, ray_path_cc]
    
    # 路径上的所有点是否在探索区域内？ (True=探索区域)
    is_passable = (exp_map[ray_path_rr, ray_path_cc] > 0)
    
    # --- 3. 查找第一个阻断点 (关键的向量化步骤) ---
    
    # 阻断条件：墙壁 OR 离开探索区域
    is_blocked = is_wall.type(torch.bool) | (~is_passable)
    
    # 查找第一个阻断点（从路径索引 1 开始，即起点之后的第一个点）
    # np.argmax 返回第一个 True 的索引。如果全部为 False，则返回 0。
    # 必须从索引 1 开始检查，起点 T_j 永远不应被视为阻断点。
    
    # 创建一个辅助数组，排除起点
    is_blocked_after_start = is_blocked[1:] 
    
    first_block_idx_relative = np.argmax(is_blocked_after_start) 
    
    # 如果 np.argmax 返回 0 且第一个点不是阻断点，说明没有阻断点
    if first_block_idx_relative == 0 and not is_blocked_after_start[0]:
        # 路径上没有阻断点 (光线射到了终点)
        final_idx = len(ray_path_rr) 
    else:
        # 找到了第一个阻断点。索引需要加 1，因为我们排除了起点
        final_idx = first_block_idx_relative + 1
    
    
    # --- 4. 标记可见性 ---
    
    # 仅考虑从起点到第一个阻断点之间的路径 (排除阻断点本身)
    visible_rr = ray_path_rr[1:final_idx]
    visible_cc = ray_path_cc[1:final_idx]
    
    # 必须确保这些点确实在探索区域内（因为 final_idx 可能是墙壁或探索边界）
    # 在这个阶段，我们只标记那些被确定可见的探索区域点
    
    # 标记：NumPy 数组的赋值操作是向量化的
    visibility_mask[visible_rr, visible_cc] = True
    return visibility_mask
    
def get_visibility_mask_reverse(object_map: np.ndarray, exp_map: np.ndarray, obs_map: np.ndarray, angle_step: float = 4.0) -> np.ndarray:
    
    H, W = object_map.shape
    visibility_mask = np.zeros((H, W), dtype=bool)

    # 1. 障碍物和目标边缘的准备 (Set查找更快)
    
    target_mask = object_map
    if target_mask.sum() == 0:
        return torch.from_numpy(visibility_mask)
    eroded_mask = binary_erosion(target_mask, footprint=np.ones((3, 3)))
    edge_mask = target_mask & (~eroded_mask)
    edge_points = np.where(edge_mask)
    if len(edge_points[0]) in range(200):
        tgt_step = 2 
    elif len(edge_points[0]) in range(100, 200):
        tgt_step = 3
    else:
        tgt_step = 5
    target_points: Set[Tuple[int, int]] = set(zip(edge_points[0][::tgt_step], edge_points[1][::tgt_step]))
    
    wall_points: Set[Tuple[int, int]] = set(zip(*np.where(obs_map > 0)))
    
    # 2. 迭代目标边缘点作为发射源
    s_angle=0
    for T_j in target_points:
        T_r, T_c = T_j
        
        # 3. 遍历 360 度方向
        for angle_deg in np.arange(s_angle%int(angle_step), 360, angle_step):
            angle_rad = math.radians(angle_deg)
            
            # 计算光线的最大终点 (地图边缘)
            # DDA/Bresenham 通常需要一个终点来确定方向和步长
            # 这里的终点是基于地图大小和角度计算出的边界点
            
            # 这是一个简化的边界计算，实际需要更鲁棒的边界查找
            max_dist = max(H, W)
            End_r = T_r + int(max_dist * math.sin(angle_rad))
            End_c = T_c + int(max_dist * math.cos(angle_rad))
            
            # 裁剪终点到地图边界内 (简化处理)
            End_r = np.clip(End_r, 0, H - 1)
            End_c = np.clip(End_c, 0, W - 1)
            
            # 获取光线路径 (从目标点 T_j 向外)
            ray_path_rr, ray_path_cc = line(T_j[0], T_j[1], End_r, End_c)
            step = 2 if len(ray_path_rr) > 1 else 1
            ray_path_rr, ray_path_cc = ray_path_rr[::step], ray_path_cc[::step]
            visibility_mask = check_ray_vectorized(ray_path_rr, ray_path_cc, H, W, exp_map, obs_map, visibility_mask)
        s_angle += 1
        
    #  最终裁剪：确保只返回探索区域内的结果
    final_mask = torch.from_numpy(visibility_mask) * exp_map
    
    return final_mask

def get_visible_regions(sem_maps, exp_maps, obs_maps, update):
    
    vis_maps = [] 
    for e in range(sem_maps.shape[0]):
        visible_map = get_visibility_mask_reverse(sem_maps[e], exp_maps[e], obs_maps[e], update[e])
        vis_maps.append(visible_map)
    return torch.stack(vis_maps)

def process_single_map(index: int, sem_maps: np.ndarray, exp_maps: np.ndarray, obs_maps: np.ndarray):
    """
    处理单个地图的可见性计算，并返回结果的 Tensor。
    这个函数在每个独立的进程中执行。
    """
    # 确保 get_visibility_mask_reverse 能够访问全局或传入的地图数组
    visible_map = get_visibility_mask_reverse(sem_maps[index], exp_maps[index], obs_maps[index])
    return torch.from_numpy(visible_map)

def get_visible_regions_multi(sem_maps, exp_maps, obs_maps):
    num_maps = sem_maps.shape[0]
    # 确定要使用的 CPU 核心数 (通常使用全部核心或核心数-1)
    num_processes = min(num_maps, os.cpu_count() or 1)
    
    with Pool(processes=num_processes) as pool:
        task_args = [(e, sem_maps, exp_maps, obs_maps) for e in range(num_maps)]
        results_list: List[torch.Tensor] = pool.starmap(process_single_map, task_args)
    
    return torch.stack(results_list)

def get_visibility_mask(sem_map: np.ndarray, exp_map: np.ndarray, obs_map: np.ndarray) -> np.ndarray:
        """
        计算探索区域中哪些点可以看到目标物体（绿色区域）的可见性掩码。
        """
        H, W = sem_map.shape
        visibility_mask = np.zeros((H, W), dtype=bool)

        # 1. 识别并集合所有障碍物、目标和探索点
        # 目标物： 只取边缘
        target_mask = sem_map < 5
        if target_mask.sum() == 0:
            return visibility_mask
        eroded_mask = binary_erosion(target_mask, footprint=np.ones((3, 3)))
        edge_mask = target_mask & (~eroded_mask)
        target_points: Set[Tuple[int, int]] = set(zip(*np.where(edge_mask)))
        
        wall_points: Set[Tuple[int, int]] = set(zip(*np.where(obs_map > 0)))
        
        # 集合所有阻挡物 (墙壁)
        # 目标物体本身也是光线的终点，而非透明阻挡物，因此在路径检查时只检查墙壁
        blockers = wall_points

        # 2. 遍历探索区域 (浅灰色像素点)
        exploration_pts = np.where(exp_map > 0)
        exp_sparse_pts = exploration_pts[0][::5], exploration_pts[1][::5]
        exploration_points = set(zip(*(exp_sparse_pts)))
        # exploration_points = set(zip(*np.where(exp_map > 0)))
        exp_sparse_values = []
        
        for r_exp, c_exp in zip(exp_sparse_pts[0], exp_sparse_pts[1]):
            P_i = (r_exp, c_exp)
            is_visible = False
            
            # 3. 对每个探索点，检查其与每个目标点之间的连线
            for T_j in target_points:
                
                # 如果探索点就是目标点 (不应发生)
                if P_i == T_j:
                    is_visible = True
                    break
                
                # 获取从 P_i 到 T_j 的像素路径
                # ray_path = bresenham_line(P_i, T_j)
                
                ray_path_rr, ray_path_cc = line(P_i[0], P_i[1], T_j[0], T_j[1])
                
                # 4. 检查光线是否被墙壁阻挡
                blocked = False
                # 遍历路径上的中间点（排除起点 P_i 和终点 T_j）
                # range(1, len(ray_path) - 1)
                for k in range(1, len(ray_path_cc)):
                    
                    # 检查中间像素 P_k 是否为墙壁
                    if (ray_path_rr[k], ray_path_cc[k]) in wall_points: 
                        blocked = True
                        break
                
                # 5. 标记可见性
                if not blocked:
                    is_visible = True
                    break # 只要能看到一个目标点，就标记为可见，并停止检查其他目标点

            if is_visible:
                visibility_mask[r_exp, c_exp] = True
                exp_sparse_values.append(1.)
            else:
                exp_sparse_values.append(0.)
                
        dense_pts = np.array(exploration_pts).T
        sparse_pts = np.stack([*exp_sparse_pts]).T
        sparse_values = np.array(exp_sparse_values)
        
        dense_scores = griddata(
            points=sparse_pts,
            values=sparse_values,
            xi=dense_pts,
            method='nearest'
        )
        
        visibility_mask[exploration_pts[0], exploration_pts[1]] = dense_scores >0.5
        return visibility_mask