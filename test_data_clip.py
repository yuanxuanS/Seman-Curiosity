import clip
from main_active_cam import clip_score
from src.finetune.dataset_utils import get_loader, SampleLoader
import torch

device = torch.device("cuda:0")   
clip_model, preprocess = clip.load("ViT-L/14", device=device)


# 读取数据集
data_pth = "/home/wpp/Seman-Curiosity/exps/dump/active_cam_eval_relost/episodes_data"

sampler = SampleLoader(data_pth)
inputs = sampler.get_env_episode_and_steps_dense_list()  

scores = []
for env, episode, step in zip(inputs[0], inputs[1], inputs[2]):
    sample_data = sampler.get_sample_multimodality(env, episode, step, ["bbsgt", "rgb", "depth"])
    rgb = sample_data['rgb'].data
    instance = sample_data['bbsgt'].data
    cls_id = instance.pred_classes
    # 
    pil_img = Image.fromarray()
    image = preprocess(pil_img).to(device)
    
    goal_idxs = []
    score = clip_score(image, goal_idxs)
    scores.append(score)
print()