import os
from dataclasses import dataclass


def _get_info_from_string(path, info, split_symbol="_"):

    filename = os.path.split(os.path.splitext(path)[0])[1]
    return filename[filename.find(info) :].split(split_symbol)[1]

def _get_info_from_string_withend(path, info, next_str="", split_symbol="_"):

    filename = os.path.split(os.path.splitext(path)[0])[1]
    start = filename[filename.find(info) :]
    start_pos = len(start.split(split_symbol)[0])
    end_pos = start.find(next_str)
    result = start[start_pos+1:end_pos-1]
    return result

@dataclass
class SenseInfo:
    """Class for keeping track of an item in inventory."""

    base_path: str
    mod: str
    episode: int = 0
    camera_id: int = 0
    step: int = 0

    def get_path(self) -> str:
        return os.path.join(
            self.base_path,
            f"episode_{self.episode:06d}_step_{self.step:05d}_modality_{self.mod}_id_{self.camera_id}.npy",
        )
        
def get_sense_info(path) -> str:

    base_path = os.path.dirname(path)

    episode = int(_get_info_from_string(path, "episode"))
    mod = _get_info_from_string_withend(path, "modality", next_str="id")
    idx = int(_get_info_from_string(path, "id"))
    step = int(_get_info_from_string(path, "step"))
    return SenseInfo(base_path, mod, episode, idx, step)
