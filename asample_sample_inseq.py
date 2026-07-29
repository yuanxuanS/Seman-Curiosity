import pickle
from src.finetune.dataset_utils import get_loader, SampleLoader

data_pth = "./exps/dump/sequencev2_wotrjR_eval/episodes_data"

sampler = SampleLoader(data_pth, glbstep=True)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


with open("./asample_straight_indices_sequencev2_wotrajR.pkl", "rb") as f:
    glb_frames_indices = pickle.load(f)

# num_every_scene = 

num_scene_cnt = {0:0, 1:0, 2:0, 3:0, 4:0}
# for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):
env_sample_indices = {0:[], 1:[], 2:[], 3:[], 4:[]}

# 保留指定数量样本
num_every_scene = 1820
sequence_sample_indices = []
for env, data in glb_frames_indices.items():
    for episode, data_epi in data.items():
        for glbstep, frames in data_epi.items():
            
            # frames = set([track.frames for track in tracks])
            for f in frames:
                env_sample_indices[env].append([episode, glbstep, f])

    # print(env_sample_indices[env])
    
cnt = 0
for e, samples in env_sample_indices.items():   
    for s in samples:   
        if cnt > num_every_scene:
            cnt = 0
            break
        epi, glb, f_ = s
        sequence_sample_indices.append([e, epi, glb, f_])
        cnt +=1 

print(f"总采样数 {len(sequence_sample_indices)},")
            

with open("./asample_straight_sampled_sequencev2_wotrajR_seq.pkl", "wb") as f:
    pickle.dump(sequence_sample_indices, f)
    
    
