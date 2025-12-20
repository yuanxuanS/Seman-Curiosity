import numpy as np
from scipy.spatial.transform import Rotation as R
import math

# --- 1. 定义输入参数 ---

# 初始向量 v = (0, 0, 1)
v_initial = np.array([0, 0, 1])

# 旋转轴 a = (0, 1, 0) (即 Y 轴)
axis = np.array([0, 1, 0])

# 旋转角度 (度数)
degree = 90  # 例如，绕Y轴旋转 90 度
# 转换为弧度
angle_rad = np.deg2rad(degree)

# --- 2. 转换为四元数（轴角表示到四元数） ---

# 四元数 R 的构造函数 R.from_rotvec() 接受“旋转向量”
# 旋转向量 = 旋转轴 * 角度(弧度)
# 注意：SciPy 的 Rotation 类内部使用的是四元数
rotation_vector = axis * angle_rad

# 创建表示旋转的四元数对象 r
r = R.from_rotvec(rotation_vector)

# --- 3. 获取四元数表示 ---

# 将四元数对象转换为 numpy 数组 (x, y, z, w) 格式
# 注意 SciPy 的格式是 (x, y, z, w)，w 是标量部分
quaternion_array_xyzw = r.as_quat()

# 将其转换为更传统的 (w, x, y, z) 格式（标量在前）
w, x, y, z = quaternion_array_xyzw[3], quaternion_array_xyzw[0], quaternion_array_xyzw[1], quaternion_array_xyzw[2]

print("--- 四元数表示 (w, x, y, z) ---")
print(f"w: {w:.4f}, x: {x:.4f}, y: {y:.4f}, z: {z:.4f}") 
# 90度绕Y轴：w=0.7071, y=0.7071

# --- 4. 使用四元数旋转向量 ---

# 使用四元数对象 r 对初始向量 v_initial 进行旋转
v_rotated = r.apply(v_initial)

print("\n--- 旋转结果 ---")
print(f"初始向量 (0, 0, 1) 绕 (0, 1, 0) 旋转 {degree}° 后的新向量:")
print(f"v' = {v_rotated.round(4)}")
# 90度旋转结果：v' = (1.0, 0.0, 0.0)

# --- 另一个例子：旋转 180 度 ---
degree_180 = 180
angle_rad_180 = np.deg2rad(degree_180)
r_180 = R.from_rotvec(axis * angle_rad_180)
v_rotated_180 = r_180.apply(v_initial)

print("\n--- 旋转 180 度结果 ---")
print(f"v'' = {v_rotated_180.round(4)}")
# 180度旋转结果：v'' = (0.0, 0.0, -1.0)