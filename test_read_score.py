import pickle
import numpy as np
# 读pkl
pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/obj_samples/"
K = 5

with open(pth + "Darden_objects_value_clip.pkl", "rb") as f:
    object_clip = pickle.load(f)

with open(pth + "Darden_objects_index.pkl", "rb") as f:
    object_index = pickle.load(f)


for key, value in object_clip.items():
    env, epi = key
    for cat, cat_value in value.items():
        for obj_id, obj_value in cat_value.items():
            step, score_arr_clip = obj_value
            print(f"===== cat{cat}, objid {obj_id}, step{step}")
            print(score_arr_clip.shape)
            # max
            # 排序并获取前 K 个最大值的索引
            sorted_indices = np.argsort(score_arr_clip.flatten())[-K:]
            row_indices, col_indices = np.unravel_index(sorted_indices, score_arr_clip.shape)   # 从0开始
            
            # 获取对应的值和索引
            top_k_values = score_arr_clip[row_indices, col_indices]
            top_k_indices = list(zip(row_indices, col_indices))
            print(f"max, Top-{K} 值:", top_k_values)
            print(f"max, Top-{K} 索引(dis, azimuth):", top_k_indices)
            
            # min
            # 排序并获取前 K 个最大值的索引
            score_arr_clip_ = score_arr_clip.copy()
            valid = score_arr_clip_ > 1e-5
            score_arr_clip_[~valid] = 1e4
            sorted_indices = np.argsort(score_arr_clip_.flatten())[:K]
            row_indices, col_indices = np.unravel_index(sorted_indices, score_arr_clip.shape)   # 从0开始
            
            # 获取对应的值和索引
            top_k_values = score_arr_clip[row_indices, col_indices]
            top_k_indices = list(zip(row_indices, col_indices))
            print(f"min, Top-{K} 值:", top_k_values)
            print(f"min, Top-{K} 索引(dis, azimuth):", top_k_indices)
            
            # 获取对应索引
            obj_idx = object_index[(env, epi)][cat][obj_id]
            for step, info in obj_idx.items():
                dis_idx, azimuth_idx = info
                print(f" env{env} epi{epi} cat {cat} obj_id{obj_id} step{step}: dis {dis_idx} azimuth {azimuth_idx}") 
                



                               
