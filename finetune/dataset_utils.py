from .sensors_utils import _get_info_from_string, \
                        _get_info_from_string_withend
                        # get_sense_info
from .sensors_data import MODALITY_SENSE      
import multiprocessing

import numpy as np

def save_obs(exp_path, episode_id, observations, timestamp):

    paths = []
    for camera_id, camera_obs in enumerate(observations):
        for modality, data in camera_obs.items():
            saved_path = _save_data(
                exp_path,
                int(episode_id),
                modality,
                int(camera_id),
                int(timestamp),
                data,
            )
            paths.append(saved_path)
    return paths

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
        self._load_paths(exp_path, samples_path)
        
    def _load_paths(self, load_path):
        samples_paths = sorted(glob.glob(load_path + '/*.npy'))

        paths = {}
        # 所有数据的episode等，相同episode、step包括了多个模态，所以一定会有重复的 TODO： 改成set？
        episode_list = [int(_get_info_from_string(s, "episode")) for s in samples_paths]
        mod_list = [_get_info_from_string_withend(s, "modality", next_str="id") for s in samples_paths]
        idx_list = [int(_get_info_from_string(s, "id")) for s in samples_paths]
        steps_list = [int(_get_info_from_string(s, "step")) for s in samples_paths]
        
        for sample_path, episode_id, input_id, mod, step in zip(
            samples_paths, episode_list, idx_list, mod_list, steps_list
        ):
            if episode_id not in paths:
                paths[episode_id] = {}
            if input_id not in paths[episode_id]:
                paths[episode_id][input_id] = {}
            if mod not in paths[episode_id][input_id]:
                paths[episode_id][input_id][mod] = {}

            paths[episode_id][input_id][mod][step] = sample_path

        self.paths = paths
        self.episode_list = np.array(episode_list)
        self.steps_list = np.array(steps_list)
        
    def __len__(self):
        pass
    
    @staticmethod
    def _load_data(path: str):
        mod = _get_info_from_string_withend(path, "modality", next_str="id")
        return MODALITY_SENSE[mod].load(path)
    
    def get_sample(self, episode, input, mod, step):
        try:
            data_path = self.paths[epsiode][input][mod][step]
            return SampleLoader._load_data(data_path)
        except Exception as ex:
            raise Exception(f"{episode}, {input}, {mod}, {step}")

    def get_episode_and_steps_dense_list(self, filter_episodes=None):
        mask = _mask_more_n(self.steps_list, 1) # 连续step相同的mask掉，去重复

        if filter_episodes is not None:
            mask_episodes = np.array(
                [li in filter_episodes for li in self.episode_list]
            )
            mask *= mask_episodes

        return self.episode_list[mask], self.steps_list[mask]
    
    def get_sample_multimodality(self, episode_id, id_camera, modalities, step):
        results = {}
        for mod in modalities:
            data = self.get_sample(episode_id, id_camera, mod, step)
            results[mod] = data
        return results

def dict_helper_collate(batch):
    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]

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