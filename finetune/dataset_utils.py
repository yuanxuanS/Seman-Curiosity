from finetune.sensors_utils import _get_info_from_string, \
                        _get_info_from_string_withend
                        # get_sense_info
from finetune.sensors_data import MODALITY_SENSE      
import multiprocessing

import numpy as np
import glob
import torch
from finetune.utils.train_helpers import dict_helper_collate
from detectron2.structures.boxes import BoxMode
import pycocotools.mask as mask_util


def save_obs(exp_path, env_id, episode_id, observations, timestamp):

    paths = []
    # for camera_id, camera_obs in enumerate(observations):
    for modality, data in observations.items():
        saved_path = _save_data(
            exp_path,
            int(env_id),
            int(episode_id),
            modality,
            int(timestamp),
            data,
        )
        paths.append(saved_path)
    return paths

def _save_data(exp_path, env_id, episode_id, modality, timestamp, data):

    path = f"{exp_path}/env_{env_id:02d}_episode_{episode_id:06d}_step_{timestamp:05d}_modality_{modality}.npy"

    np.save(
        path,
        data,
    )
    return path

def _mask_more_n(arr, n):
    '''
    将arr连续相同元素的,超出n个的mask掉
    '''
    mask = np.ones(arr.shape, np.bool_)

    current = arr[0]
    count = 0
    for idx, item in enumerate(arr):
        if item == current:
            count += 1
        else:
            current = item
            count = 1
        mask[idx] = count <= n
    return mask



class SampleLoader:
    '''
        data name type: episode_modality_id_step
        modality = MODALITY_SENSE
    '''
    def __init__(self, exp_path, samples_path=None):
        self._load_paths(exp_path)
        
    def _load_paths(self, load_path):
        samples_paths = sorted(glob.glob(load_path + '/*.npy'))

        paths = {}
        # 所有数据的episode等，相同episode、step包括了多个模态，所以一定会有重复的 TODO： 改成set？
        env_list = [int(_get_info_from_string(s, "env")) for s in samples_paths]
        episode_list = [int(_get_info_from_string(s, "episode")) for s in samples_paths]
        steps_list = [int(_get_info_from_string(s, "step")) for s in samples_paths]
        mod_list = [_get_info_from_string(s, "modality") for s in samples_paths]

        for sample_path, env_id, episode_id, step, mod in zip(
            samples_paths, env_list, episode_list, steps_list, mod_list
        ):  
            if env_id not in paths:
                paths[env_id] = {}
            if episode_id not in paths[env_id]:
                paths[env_id][episode_id] = {}
            if step not in paths[env_id][episode_id]:
                paths[env_id][episode_id][step] = {}
            if mod not in paths[env_id][episode_id][step]:
                paths[env_id][episode_id][step][mod] = {}

            paths[env_id][episode_id][step][mod]= sample_path

        self.paths = paths
        self.env_list = np.array(env_list)
        self.episode_list = np.array(episode_list)
        self.steps_list = np.array(steps_list)
        
    def __len__(self):
        pass
    
    @staticmethod
    def _load_data(path: str):
        mod = _get_info_from_string(path, "modality")
        return MODALITY_SENSE[mod].load(path)
    
    def get_sample(self, env, episode, step, mod):
        try:
            data_path = self.paths[env][episode][step][mod]
            return SampleLoader._load_data(data_path)
        except Exception as ex:
            raise Exception(f"{env}, {episode}, {step}, {mod}")

    def get_env_episode_and_steps_dense_list(self, filter_envs=None, filter_episodes=None):
        mask = _mask_more_n(self.steps_list, 1) # 连续step相同的mask掉，去重复

        if filter_envs is not None:
            mask_envs = np.array(
                [li in filter_envs for li in self.env_list]
            )
            mask *= mask_envs
            
        if filter_episodes is not None:
            mask_episodes = np.array(
                [li in filter_episodes for li in self.episode_list]
            )
            mask *= mask_episodes

        return self.env_list[mask], self.episode_list[mask], self.steps_list[mask]
    
    def get_sample_multimodality(self, env_id, episode_id, step, modalities):
        results = {}
        for mod in modalities:
            data = self.get_sample(env_id, episode_id, step, mod)
            results[mod] = data
        return results



def get_loader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=multiprocessing.cpu_count(),
    collate_fn=dict_helper_collate,
    sampler=None,
):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )
    

def get_coco_item_dict(labels):
    instances = []

    for index, y in enumerate(labels):
        class_labels = y.gt_classes

        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                'segmentation': mask_util.encode(
                    np.asfortranarray(y.gt_masks[id_instance])
                ),
                # TODO introduce 'uncertainties': y[id_instance].gt_uncertainty_masks[0],
                'iscrowd': 0,
                'infos': y[id_instance].infos[0],
                'gt_logits': y[id_instance].gt_logits[0],
            }
            for id_instance in range(len(y))
        ]

        instances.append(annotations)

    return instances