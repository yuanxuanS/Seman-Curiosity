from src.finetune.dataset_utils import SampleLoader
import clip
import torch
from asample.constants import target_coco_categories
import argparse
import pickle
import time
from src.policy_rl.sequence_utils import stc_select_samples

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stc_algorithm",
    type=str,
    default="rewrite",
    choices=["rewrite", "legacy"],
    help="STC algorithm variant for sample selection",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=3,
    help="GPU id used for CLIP inference when CUDA is available",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Run CLIP inference on CPU even when CUDA is available",
)
parser.add_argument(
    "--group_empty_tracks",
    action="store_true",
    help="Group consecutive frames without detections into tracks in rewrite STC",
)
args = parser.parse_args()

# 加载数据
stage = 2
data_pth = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequencev2_wotrjR_eval/episodes_data"
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
    with open("./asample_straight_indices_sequencev2_wotrajR.pkl", "wb") as f:
        pickle.dump(glb_frames, f)
    raise SystemExit


else:
    glb_frames_tracks = {}
    glb_frames_sampled = []
    sample_budget = 5
    clip_target_threshold = 0.5
    mod = ["bbsgt" , "bbspred", "rgb",]     #  "depth", "position", "semantic", ]

    with open("./asample_straight_indices_sequencev2_wotrajR.pkl", "rb") as f:
        glb_frames_indices = pickle.load(f)
        
    device = "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu_id}"
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
                groups, sampled_frames = stc_select_samples(
                    frame_detections,
                    frame_rgbs,
                    budget=sample_budget,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    text=text,
                    device=device,
                    clip_target_threshold=clip_target_threshold,
                    algorithm=args.stc_algorithm,
                    group_empty_tracks=args.group_empty_tracks,
                )
                print(f"stc-{args.stc_algorithm}-{time.time() - s} ")
                
                glb_frames_tracks[env][episode][glbstep] = groups   # 存储打好分的group

                for sf in sampled_frames:
                    glb_frames_sampled.append([env, episode, glbstep, sf])
                
                print(f"add tracks of env{env} epi{episode} glbstep{glbstep}")
            # 初始化

    if args.stc_algorithm == "rewrite":
        rewrite_idx = 3 if args.group_empty_tracks else 2
    else:
        rewrite_idx = 0
    with open(f"./asample_straight_tracks_sequencev2_wotrajR_bg{sample_budget}_r{rewrite_idx}_{args.stc_algorithm}.pkl", "wb") as f:
        pickle.dump(glb_frames_tracks, f)
        
    with open(f"./asample_straight_sampled_sequencev2_wotrajR_bg{sample_budget}_r{rewrite_idx}_{args.stc_algorithm}.pkl", "wb") as f:
        pickle.dump(glb_frames_sampled, f)
    print(glb_frames_sampled)
