
import math
from typing import Tuple

import numpy as np

def compute_angle_from_a2b(a, b):
    '''
    从a转向b的角度(逆时针为正)
    
    :param a, b: x, y, z
    x向右, y向上, z向后
    '''
    det = (b[0] * a[2]- a[0] * b[2])        # 叉积：得到向量，方向决定了转向
    turn_angle = math.atan2(det, np.dot(b, a))
    return turn_angle * 180 / math.pi

def compute_angle_from_a2b_2d(a, b):
    '''
    从a转向b的角度(逆时针为正)
    
    :param a, b: x, y, z
    x向右, y向上, z向后
    '''
    det = (b[0] * a[1]- a[0] * b[1])        # 叉积：得到向量，方向决定了转向
    turn_angle = math.atan2(det, np.dot(b, a))
    return turn_angle * 180 / math.pi

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

def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative); agent向前为极坐标x正，向左为y正，从x到y逆时针为朝向增加

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading. 逆时针为正方向
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
            rho: agent坐标系的极坐标下，真实世界距离
            theta:  agent坐标系的极坐标下, agent指向goal和极坐标系夹角； 极坐标系向右x正为0，逆时针为正
    """
    rotation_matrix = get_rotation_matrix(- curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)

def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")