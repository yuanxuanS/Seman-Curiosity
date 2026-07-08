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
    add_uncertainty_or_clip_fallback,
    extract_object_tracks,
    score_boundary_missing_detections,
    score_class_changes_in_tracks,
    select_top_value_samples,
)

        



# 加载数据
stage = 2
data_pth = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequencev2_eval/episodes_data"
# "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


# 遍历每一glbstep的数据
if stage == 1:
    glb_frames = {}

    for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):
        if env not in glb_frames:
            glb_frames[env] = {}
        if episode not in glb_frames[env]:
            glb_frames[env][episode] = {}
        if glbstep not in glb_frames[env][episode]:
            glb_frames[env][episode][glbstep] = [step]
        else:
            glb_frames[env][episode][glbstep].append(step)

    print(glb_frames)
    with open("./asample_straight_indices_sequencev2.pkl", "wb") as f:
        pickle.dump(glb_frames, f)
    raise SystemExit


else:
    glb_frames_tracks = {}
    glb_frames_sampled = []
    sample_budget = 3
    clip_target_threshold = 0.5
    mod = ["bbsgt" , "bbspred", "rgb",]     #  "depth", "position", "semantic", ]

    with open("./asample_straight_indices_sequencev2.pkl", "rb") as f:
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
                groups = extract_object_tracks(frame_detections)
                print(f"1-{time.time() - s} ")
                
                s = time.time()
                groups = score_boundary_missing_detections(
                    groups, len(frame_detections), sequence_detections=frame_detections
                )
                groups = score_class_changes_in_tracks(groups)
                print(f"2-{time.time() - s} ")
                
                s = time.time()
                # 从每个track中提取最高分的图像；
                groups = add_uncertainty_or_clip_fallback(
                    clip_model,
                    preprocess,
                    text,
                    groups,
                    frame_rgbs,
                    device,
                    clip_target_threshold=clip_target_threshold,
                )
                print(f"3-{time.time() - s} ")
                
                s = time.time()
                
                glb_frames_tracks[env][episode][glbstep] = groups   # 存储打好分的group
                
                sampled_frames = select_top_value_samples(groups, budget=sample_budget)
                
                for sf in sampled_frames:
                    glb_frames_sampled.append([env, episode, glbstep, sf])
                
                print(f"add tracks of env{env} epi{episode} glbstep{glbstep}")
            # 初始化

        
        
    with open("./asample_straight_tracks_sequencev2.pkl", "wb") as f:
        pickle.dump(glb_frames_tracks, f)
        
    with open("./asample_straight_sampled_sequencev2.pkl", "wb") as f:
        pickle.dump(glb_frames_sampled, f)
    print(glb_frames_sampled)
