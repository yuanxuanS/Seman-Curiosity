import cv2
import numpy as np
from src.finetune.dataset_utils import get_loader, SampleLoader
import os

data_pth = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequence6-6-1_vis/episodes_data"
data_pth_vis = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequence6-6-1_vis/episodes_data_imgs"
os.mkdir(data_pth_vis)
# "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
# inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


fts = []
rgb_types = [
        'rgb', 
        # 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240',
        #         'rgb_270', 'rgb_300', 'rgb_330', 
                ]
depth_types = [
        'depth', 'depth_30', 'depth_60', 'depth_90', 'depth_120', 'depth_150', 'depth_180', 'depth_210', 'depth_240',
                'depth_270', 'depth_300', 'depth_330',]

# types = rgb_types + depth_types
# sample_data = sampler.get_sample_multimodality(
#         1, 1, 1, types, 1)

inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  

    #    env = 0
    #    episode = 1
    #    frame = 1
    #    glbstep = 0
for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):

        sample_data = sampler.get_sample_multimodality(
                env, episode, step, rgb_types, glbstep
        )

        for rgb_ in rgb_types:
                image = np.array(sample_data[rgb_].data, copy=True) 
                cv2.imwrite(data_pth_vis+f"/env{env}_epi{episode}_gl{glbstep}_step{step}_rgb.png", image)
    # break

# for depth_ in depth_types:
#         image = np.array(sample_data[depth_].data * 255, copy=True) 
#         cv2.imwrite(f"./pano1_{depth_}.png", image)