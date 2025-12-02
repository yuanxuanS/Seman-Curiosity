#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import numpy as np
import quaternion

import math
EPSILON = 1e-8


def angle_between_quaternions(q1: np.quaternion, q2: np.quaternion) -> float:
    r"""Returns the angle (in radians) between two quaternions. This angle will
    always be positive.
=======
    从 {q}_1 旋转到 {q}_2 所需的最小旋转角
    """
    q1_inv = np.conjugate(q1)
    dq = quaternion.as_float_array(q1_inv * q2)

    return 2 * np.arctan2(np.linalg.norm(dq[1:]), np.abs(dq[0]))


def quaternion_from_two_vectors(v0: np.array, v1: np.array) -> np.quaternion:
    r"""Computes the quaternion representation of v1 using v0 as the origin."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + EPSILON):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s))


def quaternion_to_list(q: np.quaternion):
    return q.imag.tolist() + [q.real]


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion; 
        将向量v执行quat旋转
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def agent_state_target2ref(
    ref_agent_state: Union[List, Tuple], target_agent_state: Union[List, Tuple]
) -> Tuple[np.quaternion, np.array]:
    r"""Computes the target agent_state's rotation and position representation
    with respect to the coordinate system defined by reference agent's rotation and position.
    All rotations must be in [x, y, z, w] format.

    :param ref_agent_state: reference agent_state in the format of [rotation, position].
         The rotation and position are from a common/global coordinate systems.
         They define a local coordinate system.
    :param target_agent_state: target agent_state in the format of [rotation, position].
        The rotation and position are from a common/global coordinate systems.
        and need to be transformed to the local coordinate system defined by ref_agent_state.
    """

    assert (
        len(ref_agent_state[1]) == 3
    ), "Only support Cartesian format currently."
    assert (
        len(target_agent_state[1]) == 3
    ), "Only support Cartesian format currently."

    ref_rotation, ref_position = ref_agent_state
    target_rotation, target_position = target_agent_state

    # convert to all rotation representations to np.quaternion
    if not isinstance(ref_rotation, np.quaternion):
        ref_rotation = quaternion_from_coeff(ref_rotation)
    ref_rotation = ref_rotation.normalized()

    if not isinstance(target_rotation, np.quaternion):
        target_rotation = quaternion_from_coeff(target_rotation)
    target_rotation = target_rotation.normalized()

    rotation_in_ref_coordinate = ref_rotation.inverse() * target_rotation

    position_in_ref_coordinate = quaternion_rotate_vector(
        ref_rotation.inverse(), target_position - ref_position
    )

    return (rotation_in_ref_coordinate, position_in_ref_coordinate)

def compute_heading_from_quaternion(r):
    # r的Y上，Z为负朝向，X为向右
    # r - rotation quaternion
    
    # Computes clockwise rotation about Y.
    
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    # 将向量 direction_vector 从 Agent 坐标系转换到世界坐标系
    heading_vector = quaternion_rotate_vector(r, direction_vector)     
    # 向量第二个值取负，是在朝向上的投影；
    # arctan2(y,x)函数默认从x逆时针转向y，所以取负反方向；   
    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()     # [-pi, pi]
    return phi



def compute_quaternion_from_heading(theta):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    theta - heading angle in radians --- measured clockwise from -Z to X.

    Compute quaternion that represents the corresponding clockwise rotation about Y axis.
    """
    # Real part
    q0 = math.cos(-theta / 2)
    # Imaginary part
    q = (0, math.sin(-theta / 2), 0)

    return np.quaternion(q0, *q)

def compute_pitch_from_quaternion(r: np.quaternion) -> float:
    """
    计算 Agent 坐标系r 相对于世界 XZ 平面（水平面）的俯仰角 (Pitch)。
    
    使用 numpy-quaternion 库。

    基于 Agent 坐标系定义：
    Y 轴 = 向上 (Up)
    X 轴 = 向右 (Right)
    Z 轴 = 向后 (Backward)

    Args:
        r (np.quaternion): Agent 在世界坐标系中的朝向四元数。

    Returns:
        float: 俯仰角 (Pitch)，单位为弧度。正值表示向上看，负值表示向下看。
    """
    
    ## agent朝向向量
    v_local_forward = np.array([0.0, 0.0, -1.0])
    # 将向量 v_local_forward 从 Agent 坐标系转换到世界坐标系
    v_world_forward = quaternion.rotate_vectors(r, v_local_forward)
    # 获取向量 v_world_forward 的 y 分量
    argument = v_world_forward[1]
    # 使用 np.clip 确保参数在 [-1.0, 1.0] 范围内，以应对浮点误差 
    argument = np.clip(argument, -1.0, 1.0)
    
    pitch = np.arcsin(argument)
    
    return pitch

def create_pitch_quaternion_local(pitch_angle_rad: float) -> np.quaternion:
    """
    构造绕物体局部 X 轴旋转 pitch_angle_rad 的四元数。
    
    Args:
        pitch_angle_rad: 要增加的俯仰角，单位为弧度。

    Returns:
        np.quaternion: 表示该俯仰旋转的四元数。
    """
    half_angle = pitch_angle_rad / 2
    
    # q = cos(phi/2) + sin(phi/2) * i
    # 局部 X 轴为旋转轴 (1, 0, 0)
    q_pitch = np.quaternion(
        math.cos(half_angle),  # 实部 (w)
        math.sin(half_angle),  # 虚部 i (对应 X 轴)
        0.0,                   # 虚部 j (对应 Y 轴)
        0.0                    # 虚部 k (对应 Z 轴)
    )
    return q_pitch

def apply_local_rotation(rotation_world: np.quaternion, rotation_local_delta: np.quaternion) -> np.quaternion:
    """
    计算物体在世界坐标系下的新旋转位姿。
    
    Args:
        rotation_world: 物体当前在世界坐标系中的旋转位姿 (r)。
        rotation_local_delta: 要在物体坐标系下施加的额外旋转 (dr)。
        
    Returns:
        np.quaternion: 物体在世界坐标系下的新旋转位姿 (r_new)。
    """
    
    # 局部旋转（在物体自身坐标系下施加旋转）
    # 公式: r_new = r * dr
    rotation_new = rotation_world * rotation_local_delta
    
    return rotation_new

