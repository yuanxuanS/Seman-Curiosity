from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis
import math
import numpy as np
import pickle

def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion; 
        将向量v执行quat旋转
        p' = r p r^{-1},  r=quat, p=(0, v)四元数， p'为旋转后的纯四元数
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def compute_heading_from_quaternion(r):
    """
    r的Y上，Z为负朝向，X为向右
    r - rotation quaternion
    
    Computes clockwise rotation about Y.
    返回 direction vector经过r变换后的heading方向, 以及航向角yaw, (前向逆时针转的角度)
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    # 得到direction_vector在r坐标系中的朝向；
    heading_vector = quaternion_rotate_vector(r, direction_vector) 
        
    # 计算航向角yaw： 
    # yaw = atan2(hx, -hz), 此时为从-hz到 hx的角度
    # yaw 习惯上定义为从北/-z 逆时针
    # arctan2(y,x)函数默认从x逆时针转向y，所以取负反方向；   
    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()     # [-pi, pi]
    return phi, heading_vector

def compute_heading_z_from_quaternion(r):
    """
    r的Y上，Z为负朝向，X为向右
    r - rotation quaternion
    
    Computes clockwise rotation about Y.
    返回 direction vector经过r变换后的heading方向, 以及航向角yaw, (前向逆时针转的角度)
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, 1])  # positive z vector
    # 得到direction_vector在r坐标系中的朝向；
    heading_vector = quaternion_rotate_vector(r, direction_vector) 
        
    # 计算航向角yaw： 
    # yaw = atan2(hx, hz), 此时为从hz到 hx的角度
    # yaw 习惯上定义， 逆时针为正
    # arctan2(y,x)函数默认从x逆时针转向y，所以不用取反；   
    phi = np.arctan2(heading_vector[0], heading_vector[2]).item()     # [-pi, pi]
    return phi, heading_vector

def compute_angle_from_a2b(a, b):
    '''
    从a转向b的角度(逆时针为正)
    
    :param a, b: x, y, z
    x向右, y向上, z向后
    '''
    det = (b[0] * a[2]- a[0] * b[2])        # 叉积：得到向量，方向决定了转向
    turn_angle = math.atan2(det, np.dot(b, a))
    return turn_angle * 180 / math.pi

turn_angle = 90 # degree
turn_angle = turn_angle*math.pi/ 180 
# 计算绕y轴正转angle度的heading_v_quat, 四元数,
quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

# direction vector经过quat_yaw变换，在平面投影的角度
heading, heading_v = compute_heading_z_from_quaternion(quat_yaw)
print(f"从z正逆时针转 {heading * 180 / math.pi}, 正朝向为: {heading_v}")

turned = compute_angle_from_a2b([0, 0, 1], [0, 0, -1])
print(f"from a to b {turned}")
'''
- 存每个obj的正向heading
obj_id: heading_v_quat

- 求出Vheading: compute_heading_z_from_quaternion
- 求出obj指向agent向量V
- 求出Vheading 转向 V的degree即为azimuth

 验证：
- 初始化位置，检测到target时，经过以上计算得到azimuth
- 观察RGB看azimuth对不对 ，从而验证heading定义是否正确
'''

headings = {
    'Collierville': 
        {39: quat_from_angle_axis(0, np.array([0, 1.0, 0])),  # couch
         29: quat_from_angle_axis(180, np.array([0, 1.0, 0])),  # toilet
         11: quat_from_angle_axis(90, np.array([0, 1.0, 0])), # refrigerator
             },
    'Corozal':
        {}
}
with open("./gibson_headings.pkl", 'wb') as f:
    pickle.dump(headings, f)
