import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import habitat_sim
import numpy as np
import quaternion
import torch
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_to_list,
)
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray
from torch import Tensor

from habitat_extensions import maps
from habitat_baselines.common.baseline_registry import baseline_registry

cv2 = try_cv2_import()
obs_trans_to_eq = baseline_registry.get_obs_transformer("CubeMap2Equirect")
UUIDS_EQ = ['rgbback', 'rgbdown', 'rgbfront', 'rgbright', 'rgbleft', 'rgbup']
CUBE2EQ = obs_trans_to_eq(UUIDS_EQ, (224,448))


def heading_from_quaternion(quat: quaternion.quaternion) -> float:
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    heading_vector = quaternion_rotate_vector(
        quat.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

def colorize_draw_agent_and_fit_to_height(
    info: Dict[str, Any], 
    output_height: int,
    vis_info: Dict,
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    # def _ang_dis_to_coord(
    #     ang, dis, current_position, current_heading
    # ):
    #     phi = (heading_from_quaternion(current_heading) + ang) % (2 * np.pi)
    #     x = current_position[0] - dis * np.sin(phi)
    #     z = current_position[-1] - dis * np.cos(phi)
    #     return [x, z]

    top_down_map = deepcopy(info["map"])

    if vis_info is not None:
        if 'nodes' in vis_info:
            for p in vis_info['nodes']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.NODE)
        if 'ghosts' in vis_info:
            for p in vis_info['ghosts']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.GHOST)
        # if 'teacher_ghost' in vis_info and vis_info['teacher_ghost'] is not None:
        #     maps.draw_waypoint(top_down_map, vis_info['teacher_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.TEACHER_GHOST)
        if 'predict_ghost' in vis_info:
            maps.draw_waypoint(top_down_map, vis_info['predict_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.PREDICT_GHOST)
        
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["fog_of_war_mask"]
    )
    map_agent_pos = info["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map

def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


def navigator_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    # rgb = {k: v for k, v in observations.items() if k.startswith("rgb")}
    # rgb["rgb_0"] = rgb["rgb"]
    # del rgb["rgb"]
    # rgb = [
    #     f[1]
    #     for f in sorted(rgb.items(), key=lambda f: int(f[0].split("_")[1]))
    # ]

    # rgb = [
    #     add_id_on_img(rgb[i][:, 80 : (rgb[i].shape[1] - 80), :], str(i))
    #     for i in range(len(rgb))
    # ][::-1]
    # rgb = np.concatenate(rgb[6:] + rgb[:6], axis=1).astype(np.uint8)
    # new_height = int((frame_width / rgb.shape[1]) * rgb.shape[0])
    # rgb = cv2.resize(
    #     rgb,
    #     (frame_width, new_height),
    #     interpolation=cv2.INTER_CUBIC,
    # )
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    # frame = append_text_to_image(frame, observations["instruction"]["text"])

    return frame


def planner_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    frame = cv2.copyMakeBorder(frame, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,0,0))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame = append_text_to_image(frame, observations["instruction"]["text"])

    return frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    scene_id: str,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
):
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}{v:.2f}")

    video_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )