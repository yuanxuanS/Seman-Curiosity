from habitat import config
import time
import numpy as np
import torch
import copy

from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import *
from habitat.utils.geometry_utils import *
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
import magnum as mn
import quaternion





class BaseAction(SimulatorTaskAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def step(self, **kwargs):
        """reference from habitat-sim/habitat_sim/simulator.py L251
        """
        self._sim._num_total_frames += 1

        self._sim._last_state = self._sim.get_agent(0).get_state()

        # step physics by dt=1.0 / 60.0
        step_start_Time = time.time()
        # self._sim._sim.step_physics(1.0 / 60.0)  # 这个是关于对物体施加作用力，应该与我们无关
        _previous_step_time = time.time() - step_start_Time

        sim_obs = self._sim.get_sensor_observations()
        # sim_obs["collided"] = collided

        self._sim._prev_sim_obs = sim_obs
        observations = self._sim._sensor_suite.get_observations(sim_obs)

        return observations

    def _get_body_camera_angle(self):
        """
        获取当前body和camera角度（0~360°）
        """
        agent_state = self._sim.get_agent(0).get_state()
        body_rotation = agent_state.rotation
        body_angle = compute_heading_from_quaternion(agent_state.rotation)

        sensor_name = list(agent_state.sensor_states.keys())[0]  # 第0个sensor的名字
        sensor_rotation = agent_state.sensor_states[sensor_name].rotation
        sensor_angle = compute_heading_from_quaternion(sensor_rotation)

        sensor_rotation_acoord = self._sim.agents[0]._sensors['rgb'].node.rotation
        return np.rad2deg(body_angle), np.rad2deg(sensor_angle), body_rotation, sensor_rotation, sensor_rotation_acoord

    def _get_body_camera_pitch(self):
        """
        获取当前body和camera角度（0~360°）
        sensor_rotation_acoord: sensor相对body的旋转
        """
        agent_state = self._sim.get_agent(0).get_state()
        body_rotation = agent_state.rotation
        body_pitch = compute_pitch_from_quaternion(agent_state.rotation)

        sensor_name = list(agent_state.sensor_states.keys())[0]  # 第0个sensor的名字
        sensor_rotation = agent_state.sensor_states[sensor_name].rotation
        sensor_pitch = compute_pitch_from_quaternion(sensor_rotation)

        sensor_rotation_acoord = self._sim.agents[0]._sensors['rgb'].node.rotation
        return np.rad2deg(body_pitch), np.rad2deg(sensor_pitch), body_rotation, sensor_rotation, sensor_rotation_acoord
    
    def _cal_camera_angle(self, target_turn_angle, limited_angle=90):
        '''
        target_turn_angle： 绕y轴旋转角度(逆时针)
        限制camera与body的角度不能超过一定度数
        limited_angle: 机器人的camera和body夹角不大于limited_angle (可设置0~180)
        
        '''
        # 获取当前body和camera角度
        body_angle_new, camera_angle, body_rotation_new, camera_rotation, cam_rot_acoord = self._get_body_camera_angle()
        camera_angle_tmp = -target_turn_angle + camera_angle        # heading方向是顺时针为正
        if camera_angle_tmp- body_angle_new> limited_angle:
            target_turn_angle = limited_angle + body_angle_new
        elif body_angle_new- camera_angle_tmp > limited_angle:
            target_turn_angle = -limited_angle + body_angle_new
        else:
            target_turn_angle = target_turn_angle
        camera_rotation_new = self._move_camera_horizental(cam_rot_acoord, target_turn_angle)
        # camera_angle_new = np.rad2deg(compute_heading_from_quaternion(camera_rotation_new))

        # print("action_name:", self._get_uuid())
        # print("body_angle_new:{} camera_angle_new:{} relative_angle:{} ".format(body_angle_new, camera_angle_new, (camera_angle_new - body_angle_new) % 360))
        
        
        return camera_rotation_new

    def quat_to_mnquat(self, quat):
        mnquat = mn.Quaternion(mn.Vector3(quat.x, 
                                        quat.y, 
                                        quat.z), quat.w)
        return mnquat
    
    def mnquat_to_quat(self, mnquat):
        w = mnquat.scalar
        x = mnquat.vector.x
        y = mnquat.vector.y
        z = mnquat.vector.z
        quat = np.quaternion(w, x, y, z)
        return quat
        
    def _move_camera_vertical(self, sensor_rotation, amount: float):
        if not isinstance(sensor_rotation, mn.Quaternion):
            sensor_rotation = self.quat_to_mnquat(sensor_rotation)
        
        # 对sensor_rotation根据上层(agent)坐标系的x轴旋转amount角度, 所以左乘
        sensor_rotation_new =  mn.Quaternion.rotation(
            mn.Deg(amount), mn.Vector3.x_axis()
        ) * sensor_rotation
        
        # to quaternion type
        sensor_rotation_new = self.mnquat_to_quat(sensor_rotation_new)
        return sensor_rotation_new
    
    def _move_camera_horizental(self, sensor_rotation, amount: float):
        '''
        对sensor_rotation根据上层(agent)坐标系的-y轴旋转amount角度
        '''
        if not isinstance(sensor_rotation, mn.Quaternion):
            sensor_rotation = self.quat_to_mnquat(sensor_rotation)
            
        sensor_rotation_new = sensor_rotation * mn.Quaternion.rotation(
            mn.Deg(amount), - mn.Vector3.y_axis()       # 定义绕顺时针为正，所以y轴逆向
        )
        
        # to quaternion type
        sensor_rotation_new = self.mnquat_to_quat(sensor_rotation_new)
        return sensor_rotation_new
    
    def _cal_camera_pitch(self, target_turn_angle, limited_pitch=60):
        '''
        对sensor相对agent的坐标cam_rot_acoord, 在agent坐标系中进行变换
        limited_angle: 限制camera俯仰角度不能超过一定度数
        
        '''
        body_pitch_new, camera_pitch, body_rotation_new, camera_rotation, cam_rot_acoord = self._get_body_camera_pitch()
        # limit range
        camera_pitch_tmp = target_turn_angle + camera_pitch
        if camera_pitch_tmp > limited_pitch:
            target_turn_angle = limited_pitch - camera_pitch
        elif camera_pitch_tmp < -limited_pitch:
            target_turn_angle = -limited_pitch - camera_pitch
        else:
            target_turn_angle = target_turn_angle
        # 
        camera_rot_acoord_new = self._move_camera_vertical(cam_rot_acoord, target_turn_angle)
        # camera_pitch_new = np.rad2deg(compute_pitch_from_quaternion(camera_rot_acoord_new))

        # print("action_name:", self._get_uuid())
        # print("body_angle_new:{} camera_angle_new:{} relative_angle:{} ".format(body_angle_new, camera_angle_new, (camera_angle_new - body_angle_new) % 360))
        return camera_rot_acoord_new
    
    def _implement_camera_action(self, camera_rotation_new):
        '''把camera设置到某个角度
        camera_angle_new: 需要设置到的角度（单位°）
        '''
        sensor_names = list(self._sim.agents[0]._sensors.keys())
        for sensor_name in sensor_names:
            sensor = self._sim.agents[0]._sensors[sensor_name].node
            sensor.rotation = self.quat_to_mnquat(camera_rotation_new)
        # self._sim.get_agent(0).set_state(agent_state)
        # self._sim._sim.get_agent(0).set_state(agent_state, infer_sensor_states=False)

@registry.register_task_action
class TransportAction(BaseAction):
    '''
    传送到指定地点
    '''
    def _get_uuid(self, *args, **kwargs) -> str:
        return "transport"
    
    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        
        target_loc = kwargs['tp_loc']
        pos, rot = target_loc
        pos = np.array(pos)
        rot = quaternion.from_float_array(np.array(rot))
        # self._sim.set_agent_state(pos, rot)
        agent_state = self._sim.get_agent(0).get_state()
        agent_state.position = pos
        agent_state.rotation = rot
        self._sim.get_agent(0).set_state(agent_state, infer_sensor_states=True)

        # sensor_names = list(self._sim.agents[0]._sensors.keys())
        # for sensor_name in sensor_names:
        #     sensor = self._sim.agents[0]._sensors[sensor_name].node
        #     sensor.position = pos
        #     sensor.rotation = rot
            
        observations = super().step()

        return observations
    
@registry.register_task_action
class CameraCaptureAction(BaseAction):
    '''
    相机捕获观测，认为episode结束
    '''
    def _get_uuid(self, *args, **kwargs) -> str:
        return "camera_capture"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        # collided = self._sim._sim.get_agent(0).act(HabitatSimActions.MOVE_FORWARD)
        a = self._cal_camera_angle(0)
        observations = super().step()

        return observations
    



@registry.register_task_action
class CameraLeftAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "camera_left"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False

        camera_delta = kwargs['task']._sim.habitat_config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.habitat_config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(-camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step()

        return observations


@registry.register_task_action
class CameraRightAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "camera_right"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False

        camera_delta = kwargs['task']._sim.habitat_config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.habitat_config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step()

        return observations

@registry.register_task_action
class CameraUpAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "camera_up"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False

        camera_delta = kwargs['task']._sim.habitat_config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.habitat_config.CAMERA_PITCH_LIMIT
        camera_rot_acoord_new = self._cal_camera_pitch(camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rot_acoord_new)

        observations = super().step()

        return observations

@registry.register_task_action
class CameraDownAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "camera_down"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False

        camera_delta = kwargs['task']._sim.habitat_config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.habitat_config.CAMERA_PITCH_LIMIT
        camera_rotation_new = self._cal_camera_pitch(-camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step()

        return observations
    
# def joint_action(body_action, camera_action, cfg_AC: config, time_steps, test_new_baseline = None):
#     fourAction2tenAction = {0:0, 1:1, 2:5, 3:9}

#     if cfg_AC.camera_AC == 'none':   # multiON的设置，输出4个动作
#         b_action = torch.zeros_like(body_action)
#         for i in range(len(body_action)):
#             b_action[i] = fourAction2tenAction[body_action[i].item()]
#         action = b_action
#     elif cfg_AC.body_AC == 'e2e' and cfg_AC.camera_AC == 'e2e': # e2e同时决定body和camera，输出10个动作
#         action = body_action
#     else:
#         mask = (body_action-1)>=0   # 判断是否执行found

#         if test_new_baseline is not None:
#             for idx, flag in enumerate(test_new_baseline):
#                 if flag:
#                     body_action[idx] = camera_action[idx] + 1 # only for another baseline testing

#         action = (body_action-1)*3 + camera_action + 1
#         action *= mask

#     action[time_steps < 3] = 5   # 前三步左转左看
#     body_action[time_steps < 3] = 2
#     camera_action[time_steps < 3] = 1
#     return action
