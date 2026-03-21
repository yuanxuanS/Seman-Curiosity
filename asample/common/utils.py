import numpy as np


def save_obs(exp_path, env_id, episode_id, observations,glbstep, timestamp):

    paths = []
    # for camera_id, camera_obs in enumerate(observations):
    for modality, data in observations.items():
        saved_path = _save_data(
            exp_path,
            int(env_id),
            int(episode_id),
            modality,
            int(glbstep),
            int(timestamp),
            data,
        )
        paths.append(saved_path)
    return paths

def _save_data(exp_path, env_id, episode_id, modality, glbstep, timestamp, data):

    path = f"{exp_path}/env_{env_id:02d}_episode_{episode_id:06d}_gl_{glbstep:02d}_step_{timestamp:05d}_modality_{modality}.npy"

    np.save(
        path,
        data,
    )
    return path