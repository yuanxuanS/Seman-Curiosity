import numpy as np
from typing import Tuple, Set, List
from skimage.draw import line
import time

# --- 假设的地图常量 (请根据您的实际地图值进行调整) ---
WALL_VALUE = 10     # 深灰色/墙壁
TARGET_VALUE = 50   # 绿色目标物体
PASSABLE_VALUE = 1  # 浅灰色探索区域

# --- 占位函数：您需要实现或引入高效的像素线段生成函数 ---
def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    [TODO: 实际项目中需替换为高效的 Bresenham/DDA 实现]
    返回从 start 到 end 的像素坐标列表，包括 start 和 end。
    """
    r0, c0 = start
    r1, c1 = end
    
    # 简化占位实现，确保它能用
    points = []
    num_steps = max(abs(r1 - r0), abs(c1 - c0))
    if num_steps == 0:
        return [start]
        
    for i in range(num_steps + 1):
        r = r0 + (r1 - r0) * i / num_steps
        c = c0 + (c1 - c0) * i / num_steps
        points.append((int(round(r)), int(round(c))))
        
    return points


def get_visibility_mask(sem_map: np.ndarray, exp_map: np.ndarray, obs_map: np.ndarray) -> np.ndarray:
    """
    计算探索区域中哪些点可以看到目标物体（绿色区域）的可见性掩码。
    """
    H, W = sem_map.shape
    visibility_mask = np.zeros((H, W), dtype=bool)

    # 1. 识别并集合所有障碍物、目标和探索点
    # 目标物： 只取边缘
    
    target_points: Set[Tuple[int, int]] = set(zip(*np.where(sem_map == TARGET_VALUE)))
    wall_points: Set[Tuple[int, int]] = set(zip(*np.where(obs_map == WALL_VALUE)))
    
    # 集合所有阻挡物 (墙壁)
    # 目标物体本身也是光线的终点，而非透明阻挡物，因此在路径检查时只检查墙壁
    blockers = wall_points

    # 2. 遍历探索区域 (浅灰色像素点)
    exploration_points = np.argwhere(exp_map == PASSABLE_VALUE)
    
    for r_exp, c_exp in exploration_points:
        P_i = (r_exp, c_exp)
        is_visible = False
        
        # 3. 对每个探索点，检查其与每个目标点之间的连线
        for T_j in target_points:
            
            # 如果探索点就是目标点 (不应发生)
            if P_i == T_j:
                is_visible = True
                break
            
            # 获取从 P_i 到 T_j 的像素路径
            ray_path = bresenham_line(P_i, T_j)
            
            # 4. 检查光线是否被墙壁阻挡
            blocked = False
            # 遍历路径上的中间点（排除起点 P_i 和终点 T_j）
            # range(1, len(ray_path) - 1)
            for k in ray_path[1:-1]:
                
                # 检查中间像素 P_k 是否为墙壁
                if k in wall_points: 
                    blocked = True
                    break
            
            # 5. 标记可见性
            if not blocked:
                is_visible = True
                break # 只要能看到一个目标点，就标记为可见，并停止检查其他目标点

        if is_visible:
            visibility_mask[r_exp, c_exp] = True
            
    return visibility_mask

s = time.time()
pts = bresenham_line((1,1), (200,200))
print(f"func 1 time {time.time() - s}")
# print(pts)

s = time.time()
rr, cc = line(199, 120, 200, 200)
print(f"func 2 time {time.time() - s}")

print(rr, cc)