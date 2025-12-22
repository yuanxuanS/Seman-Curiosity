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



turn_angle = 180 # degree
turn_angle = turn_angle*math.pi/ 180 
# 计算绕y轴正转angle度的heading_v_quat, 四元数,
quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

# direction vector经过quat_yaw变换，在平面投影的角度
heading, heading_v = compute_heading_z_from_quaternion(quat_yaw)
print(f"从z正逆时针转 {heading * 180 / math.pi}, 正朝向为: {heading_v}")

'''
turned = compute_angle_from_a2b([0, 0, 1], heading_v)    # 从z正转向heading为azi
- 存每个obj的正向heading

- 求出Vheading: compute_heading_z_from_quaternion
- 求出obj指向agent向量V
- 求出Vheading 转向 V的degree即为azimuth



- 先设置所有obj的朝向为0；然后观测azimuth朝向哪里，需要如何调整达到目标朝向；

'''


headings = {
    'Collierville': 
        {39: quat_from_angle_axis((-80)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
         29: quat_from_angle_axis((100)*math.pi / 180, np.array([0, 1.0, 0])),  # toilet
         11: quat_from_angle_axis((0)*math.pi / 180, np.array([0, 1.0, 0])), # refrigerator
             },
    'Corozal':
        {
            35: quat_from_angle_axis((-170)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            34: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            85: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # refri
            52: quat_from_angle_axis((0)*math.pi / 180, np.array([0, 1.0, 0])),  # bed
            60: quat_from_angle_axis((180)*math.pi / 180, np.array([0, 1.0, 0])),  # toliet
            59: quat_from_angle_axis((90)*math.pi / 180, np.array([0, 1.0, 0])),  # toliet
        },
    'Darden':
        {
            58: quat_from_angle_axis((0)*math.pi / 180, np.array([0, 1.0, 0])),  # toilet
            52: quat_from_angle_axis((90)*math.pi / 180, np.array([0, 1.0, 0])),  # bed
            37: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            36: quat_from_angle_axis((-45)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            35: quat_from_angle_axis((0)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            72: quat_from_angle_axis((180)*math.pi / 180, np.array([0, 1.0, 0])),  # refri
            74: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # refri
            75: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # refri
        },
        'Markleeville':{
            52: quat_from_angle_axis((90)*math.pi / 180, np.array([0, 1.0, 0])),  # refri
            39: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # toliet
            34: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # bed
            23: quat_from_angle_axis((135)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
        },
        'Wiconisco':{
            26: quat_from_angle_axis((-90)*math.pi / 180, np.array([0, 1.0, 0])),  # couch
            53: quat_from_angle_axis((180)*math.pi / 180, np.array([0, 1.0, 0])), # toliet
            68: quat_from_angle_axis((45)*math.pi / 180, np.array([0, 1.0, 0])), # refri
    }
}
# with open("./gibson_headings.pkl", 'wb') as f:
#     pickle.dump(headings, f)

obj_loc_real = {
    'Collierville': 
        {
            39: (-1.1099999999999994, 0.0, 0.7149999999999999),
            29: (1.0150000000000006, 0.0, -6.034999999999999),
            11: (-0.034999999999999254, 0.0, -5.859999999999999),
        },
    'Corozal':
        {
            35: (-1.540000000000001, 0.0, -3.744999999999999), # couch
            34: (0.8100000000000005, 0.0, -6.219999999999999),  # couch
            85: (-0.2900000000000009, 0.0, 1.004999999999999),  # refri
            52: (-0.7400000000000002, 0.0, -11.594999999999999),    # bed
            60: (-6.965, 0.0, -11.82),  # toliet
            59: (-5.84, 0.0, -9.62),    # toliet
        },
    'Darden':
        {
            58: (-4.530000000000001, 0.0, -3.615),  # toilet
            52: (-8.430000000000001, 0.0, -2.365), # bed
            37: (-10.305000000000001, 0.0, -2.165), # couch
            36: (-10.280000000000001, 0.0, 1.085), # couch
            35: (-1.4050000000000011, 0.0, -3.415), # couch
            72: (-0.08000000000000185, 0.0, 2.335), # refri
            74: (-5.1800000000000015, 0.0, -3.615), # refri
            75: (-5.755000000000001, 0.0, -3.04), # refri
        },
    'Markleeville':{
            52: (2.085, 0.0, -4.845000000000001), # refrigerator
            39: (0.96, 0.0, -5.42),     # toliet
            34: (0.2599999999999998, 0.0, -0.14500000000000046),    # bed
            23: (-1.3900000000000001, 0.0, 1.5299999999999994)  # couch
        },
    'Wiconisco':{
        26: (5.734999999999999, 0.0, -2.705),   # couch
        53: (-6.215, 0.0, -4.055),  # toliet
        68: (3.51, 0.0, -11.18), # refri
    }
}

# with open("./gibson_objects_loc.pkl", 'wb') as f:
#     pickle.dump(obj_loc_real, f)

obj_loc_real = {
    'Collierville': 
        
            [
                [39,(-1.1099999999999994, 0.0, 0.7149999999999999), 'couch', 0],
                [29,(1.0150000000000006, 0.0, -6.034999999999998), 'toilet', 1],
                [11,(-0.034, 1.2345678e-16, -5.85), 'refrigerator', 2],
                [None, (-1.6849999999999996, 0.0, -2.4349999999999996), 'chair', 3]
            ],
        
    'Corozal':
        [
            [35,(-1.540000000000001, 0.0, -3.744999999999999), 'couch', 0], # couch
            [34,(0.8100000000000005, 0.0, -6.219999999999998), 'couch', 1],  # couch
            [85,(-0.29, 1.2345678e-16, 1.2345678e-16), 'refrigerator', 2],  # refri
            [52,(-1.54, 1.2345678e-16, -3.74), 'bed', 3],    # bed
            [60,(-6.215, 1.2345678e-16, -4.8), 'toilet', 4],  # toliet
            [59,(-5.84, 1.2345678e-16, -3.7), 'toilet', 5],    # toliet
            [None, (-8.540000000000001, 0.0, -9.42), 'chair', 6],   # chair
            [None, (-6.515000000000001, 0.0, -2.3949999999999996), 'chair', 7],   # chair
            [None, (-1.2400000000000002, 0.0, -1.5949999999999989), 'chair', 8],   # chair
            
        ],
    'Darden':
        [
            [58,(-4.530000000000001, 0.0, -3.615), 'toilet', 0],  # toilet
            [52,(-8.430000000000001, 0.0, -2.365), 'bed', 1], # bed
            [37,(-10.305, 1.2345678e-16, -2.165), 'couch', 2], # couch
            [36,(-10.28, 1.2345678e-16, 1.085), 'couch', 3], # couch
            [35,(-1.405, 1.2345678e-16, -3.415), 'couch', 4], # couch
            [72,(-0.8, 1.2345678e-16, -2.9), 'refrigerator', 5], # refri
            [74,(-5.18, 1.2345678e-16, -3.6), 'refrigerator', 6], # refri
            [75,(-5.755000000000001, 1.2345678e-16, -3.04), 'refrigerator', 7], # refri
            [None, (-14.205000000000002, 0.0, -2.74), 'chair', 8],   # chair
            [None, (-12.330000000000002, 0.0, -2.8150000000000004), 'chair', 9],   # chair
            [None, (-11.055000000000001, 0.0, 1.0350000000000001), 'chair', 10],   # chair
            [None, (-9.23, 0.0, -3.165), 'chair', 11],   # chair
            [None, (-9.255, 0.0, -1.2150000000000003), 'chair', 12],   # chair
            [None, (-12.330000000000002, 0.0, -1.5650000000000004), 'chair', 13],   # chair
            [None, (-7.4300000000000015, 0.0, 1.2850000000000001), 'chair', 14],   # chair
            [None, (-0.5050000000000026, 0.0, -0.8150000000000004), 'chair', 15],   # chair
            
        ],
    'Markleeville':
        [
            [52, (2.085, 0.0, -4.845000000000001), 'refrigerator', 0],
            [39, (0.96, 0.0, -5.42), 'toilet', 1],     # toliet
            [34, (0.2599999999999998, 0.0, -0.14500000000000046), 'bed', 2],    # bed
            [23, (-1.39, 1.2345678e-16, 1.5299999999999994), 'couch', 3],  # couch
            [None, (6.16, 0.0, -5.445), 'chair', 4],  # chair
            [None, (7.584999999999999, 0.0, -5.07), 'chair', 5],  # chair
            [None, (7.584999999999999, 0.0, 0.9550000000000001), 'chair', 6],  # chair
            [None, (8.334999999999999, 0.0, -0.2450000000000001), 'chair', 7],  # chair
        ],
    'Wiconisco':
        [
            [26, (5.734999999999999, 0.0, -2.705), 'couch', 0],   # couch
            [53, (-6.215, 0.0, -4.055), 'toilet', 1],  # toliet
            [68, (3.51, 0.0, -11.18), 'refrigerator', 2], # refri
            [None, (-5.965, 0.0, -0.6300000000000008), 'chair', 4],  # chair
            [None, (-5.99, 0.0, 1.3449999999999989), 'chair', 5],  # chair
            [None, (-1.3399999999999999, 0.0, 0.46999999999999886), 'chair', 6],  # chair
            [None, (0.41000000000000014, 0.0, 0.5949999999999989), 'chair', 7],  # chair
            [None, (2.1099999999999994, 0.0, -5.58), 'chair', 8],  # chair
            [None, (3.41, 0.0, -5.155), 'chair', 9],  # chair
            [None, (4.3100000000000005, 0.0, -9.105), 'chair', 10],   # chair
            [None, (4.584999999999999, 0.0, -6.83), 'chair', 11],  # chair
            [None, (4.484999999999999, 0.0, -5.130000000000001), 'chair', 12],  # chair
        ]
}

with open("./gibson_objects_loc2.pkl", 'wb') as f:
    pickle.dump(obj_loc_real, f)
    
# t= np.quaternion(-0.22018953669540486, 0.0, 0.9754571071707168, 0.0)
# an = compute_heading_z_from_quaternion(t)[0]
# print(an*180/math.pi)

def compute_angle_from_a2b_2d(a, b):
    '''
    从a转向b的角度(逆时针为正)
    
    :param a, b: x, y, z
    x向右, y向上, z向后
    '''
    det = (b[0] * a[1]- a[0] * b[1])        # 叉积：得到向量，方向决定了转向
    turn_angle = math.atan2(det, np.dot(b, a))
    return turn_angle * 180 / math.pi

v1 = [0, 1]
v2 = [1, 1.732]
print(compute_angle_from_a2b_2d(v1, v2))
