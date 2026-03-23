import cv2
import numpy as np
from src.finetune.dataset_utils import get_loader, SampleLoader

data_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/test_test/episodes_data"
# "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
# inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


fts = []
rgb_types = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240',
                'rgb_270', 'rgb_300', 'rgb_330']

sample_data = sampler.get_sample_multimodality(
        0, 1, 5, rgb_types, 0)

for rgb_ in rgb_types:
    image = np.array(sample_data[rgb_].data, copy=True) 
    cv2.imwrite(f"./pano_{rgb_}.png", image)
    # break