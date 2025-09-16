import numpy as np
from src.finetune.dataset_utils import get_loader, SampleLoader
import cv2
import os

base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env5/"
new_save_dir = base_dir + 'masked_obj/'
if not os.path.exists(new_save_dir):
    os.mkdir(new_save_dir)
    
data_pth = base_dir + "episodes_data"
multi_obj_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env5/multiobj.npz"
# load multi obj image
data = np.load(multi_obj_pth)['data']
sampler = SampleLoader(data_pth)
# iter every obj, save mask
for idx in data:
    env, episode, step = idx
    sample_data = sampler.get_sample_multimodality(env, episode, step, ["bbsgt", "rgb"])
    rgb = sample_data['rgb'].data
    instance = sample_data['bbsgt'].data
    
    for i in range(len(instance.pred_classes)):
        cls_id = instance.pred_classes[i]
        mask_ = (instance.pred_masks)[i][:, :, None]
        masked_obj = rgb * mask_.cpu().numpy()
        cv2.imwrite(new_save_dir + f"epi{str(episode)}_env{str(env)}_step{str(step)}_num{str(i)}_cls{str(cls_id.item())}.png", masked_obj)
        # 名字里带顺序id + 类别id 来区分
