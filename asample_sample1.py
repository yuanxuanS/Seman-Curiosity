from src.finetune.dataset_utils import get_loader, SampleLoader
from src.policy_rl.agents.utils.detect_utils import box_iou_calc
import numpy as np
from PIL import Image
import clip
import torch
from asample.constants import target_coco_categories
import pickle
import time
from src.policy_rl.sequence_utils import (
    get_all_sample_score,
    get_specify_samples,
    aggre_score_in_obj_tracks,
    score_tracks,
    group_by_object_and_score,
)

        



# 加载数据
data_pth = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequence_coverage_eval/episodes_data"
# "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


# 遍历每一glbstep的数据
# glb_frames = {}
# glb_frame = []

# for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):
    
#     if env not in glb_frames:
#         glb_frames[env] = {}
#     if episode not in glb_frames[env]:
#         glb_frames[env][episode] = {}
#     if glbstep not in glb_frames[env][episode]:
#         glb_frames[env][episode][glbstep] = [step]
#     else:
#         glb_frames[env][episode][glbstep].append(step)
# print(glb_frames)
# with open("./asample_straight_indices_sequence_coverage.pkl", "wb") as f:
#     pickle.dump(glb_frames, f)



glb_frames_tracks = {}
glb_frames_sampled = []
mod = ["bbsgt" , "bbspred", "rgb",]     #  "depth", "position", "semantic", ]

with open("./asample_straight_indices_sequence_coverage.pkl", "rb") as f:
    glb_frames_indices = pickle.load(f)
    
device = "cuda:3" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
text_str = [key for key in target_coco_categories.keys()]
text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)


#组合每一track并进行采样
for env, env_data in glb_frames_indices.items():
    
    if env not in glb_frames_tracks:
        glb_frames_tracks[env] = {}
        
    for episode, epi_data in env_data.items():
        
        if episode not in glb_frames_tracks[env]:
            glb_frames_tracks[env][episode] = {}
            
        for glbstep, frames in epi_data.items():
            
            
            glb_datas = []
            for f in sorted(frames):
                sample_data = sampler.get_sample_multimodality(
                    env, episode, f, mod, glbstep)
                glb_datas.append(sample_data)
                
            # 对每一glbstep的数据进行分组；同一位置且统一类别为同一组
            
            frame_detections = [data['bbspred'].data for data in glb_datas]
            # print(frame_detections[0].data)
            # break
            frame_rgbs = [data['rgb'].data for data in glb_datas]
            
            s = time.time()
            groups = group_by_object_and_score(frame_detections)
            print(f"1-{time.time() - s} ")
            
            s = time.time()
            groups = score_tracks(groups, frame_detections)
            print(f"2-{time.time() - s} ")
            
            s = time.time()
            groups = aggre_score_in_obj_tracks(groups, mode="track")
            print(f"3-{time.time() - s} ")
            
            s = time.time()
            # 从每个track中提取最高分的图像；
            groups = get_all_sample_score(clip_model, preprocess, text, groups, frame_rgbs, device)
            print(f"4-{time.time() - s} ")
            
            s = time.time()
            
            glb_frames_tracks[env][episode][glbstep] = groups   # 存储打好分的group
            
            sampled_frames = get_specify_samples(groups)
            
            for sf in sampled_frames:
                glb_frames_sampled.append([env, episode, glbstep, sf])
            
            print(f"add tracks of env{env} epi{episode} glbstep{glbstep}")
        # 初始化

    
    
with open("./asample_straight_tracks_sequence_coverage.pkl", "wb") as f:
    pickle.dump(glb_frames_tracks, f)
    
with open("./asample_straight_sampled_sequence_coverage.pkl", "wb") as f:
    pickle.dump(glb_frames_sampled, f)
print(glb_frames_sampled)

