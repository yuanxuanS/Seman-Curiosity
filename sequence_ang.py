import numpy as np

def map_radians_to_intervals(rad_array):
    # 1. 将弧度转换为角度 [-180, 180]
    angles = np.degrees(rad_array)
    print(angles)
    # 2. 处理角度范围，使其与你的区间定义一致
    # 你的区间从 -30 开始，到 330 结束。
    # 我们将所有角度映射到 [-30, 330) 范围内
    angles_shifted = np.mod(angles + 30, 360) - 30
    print(angles_shifted)
    # 3. 定义区间边界 (左闭右开)
    # 区间 0: [-30, 30), 1: [30, 90), 2: [90, 150), 3: [150, 240), 4: [240, 330)
    # 注意：你只列到了区间 4，剩下的 330 到 -30 实际上是第 6 个区间（区间 5）
    bins = [-30, 30, 90, 150, 240, 330]
    
    # 4. 使用 digitize 进行分类 (返回索引 1-5，减 1 变为 0-4)
    indices = np.digitize(angles_shifted, bins) - 1
    
    # 处理超出 330 的部分归为区间 5 (或者根据你的需求处理)
    indices[indices == 5] = 5 
    
    return indices

# 测试数据
test_rad = np.array([-np.pi, -np.pi/6, 0, np.pi/2, np.pi])
result = map_radians_to_intervals(test_rad)
print(result)
for r, idx in zip(test_rad, result):
    print(f"弧度: {r:6.2f} | 角度: {np.degrees(r):7.2f}° | 所属区间: {idx}")