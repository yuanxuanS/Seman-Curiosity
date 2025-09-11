import pickle
from src.finetune.dataset_utils import get_loader, SampleLoader
import shutil
import random

dir = "/data1/wpp_data/data/vsqf_test_val5_orig/"
scene = "Markleeville" # 2:"Darden" # 1:"Corozal" #0:"Collierville" #4: "Wiconisco"
dst_env_id = 3      # 环境id
dst_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/vsqf_test_val5/data/"
sample_num = 1000 + 56

def conditions(step):
    if step > 240:
        return True
    return False
    # Darden : 排除 2736-3059
    # Wico: 排除 3260-3476
    # Mark: 排除240以前， 
    # return True

with open(dir+f"/{scene}_objects.pkl", "rb") as f:
    object_index = pickle.load(f)

sampler = SampleLoader(dir + "/data/"+scene)

data_lst = []
for key, value in object_index.items():
    env, epi = key
    for cat, cat_obj in value.items():
        if len(cat_obj)> 0:
            for obj_id, v_ in cat_obj.items():
                # for step, index in v_.items():
                scene_obj_id, obj_samples = v_ 
                for dis_k, dis_v in obj_samples.items():
                    for yaw_k, step in dis_v.items():
                        if conditions(step):
                            instances = sampler.get_sample(env, epi, step, "bbsgt").get_bbs_as_gt()
                            if len(instances) > 0:
                                data_lst.append((env, epi, step, cat, scene_obj_id))
print(len(data_lst))
random.shuffle(data_lst)
# print(data_lst)

# read data

modalities = ['bbsgt', 'compass', 'depth', 'gps', 'position', 'rgb', 'semantic']

cls_static = {0:0, 1:0, 3:0, 4:0, 9:0}

for data in data_lst[:sample_num]:
    env, epi, step, cat, scene_obj_id = data
    cls_static[cat] += 1
    for mod in modalities:
        src_file = f"{dir}/data/{scene}/env_{env:02d}_episode_{epi:06d}_step_{step:05d}_modality_{mod}.npy"
        dst_file = f"{dst_dir}/env_{dst_env_id:02d}_episode_{epi:06d}_step_{step:05d}_modality_{mod}.npy"
        shutil.copy(src_file, dst_file)
        print(f"cp {src_file} to {dst_file}")
print(f" scene {scene} class {cls_static}")
print(f" scene {scene}, sample num : {len(data_lst[:sample_num])}")