import clip
from draw_datasets import load_data_imgs
from vqf_dataset import real_obj_statistics
import torch
from src.finetune.sensors_data import AgentPoseSense, BBSense
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from copy import deepcopy
from PIL import Image
from src.vqf_constants import clsid_name_maps
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pickle

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


# 读取所有样本
note = "vsqf_v3_1_eval"
dataset_path = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/vsqf_v3_1_eval/episodes_data"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/vsqf_v1_eval_re/episodes_data"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/vsqf_v1_3_eval/episodes_data"


test_loader = load_data_imgs(dataset_path, None)

samples = []
for batch_idx, batch in enumerate(test_loader):
    for idx, x in enumerate(batch):
        remap = BBSense.REMAP
        metadata = MetadataCatalog.get('coco_2017_val')
        
        y = deepcopy(x['instances'])
        # y = deepcopy(x['instances_pred'])
        if len(x['instances']) == 0:
            print(f"no ins in env {x['env']} epi{x['episode']} step {x['step']}")
        
        img = x['image'].permute(1, 2, 0).cpu().numpy()

        # 多个物体，选分数最低的代表该样本
        
        min_score = 1e4
        min_class = ""
        for p in y.gt_classes:
            remap_id = remap[p.item()]
            remap_name = clsid_name_maps[remap_id]
            

            # clip打分每一张图
            text_str = [remap_name]  #'cabinet.']        # 'cabinet']# 
            text = clip.tokenize([f"a photo contains a {i}." for i in text_str]).to(device)

            pil_img = Image.fromarray(img.astype(np.uint8))
            image = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                score = logits_per_text.item()
            # 分数归一化
            score_norm = (score - real_obj_statistics[remap_name][0]) / (real_obj_statistics[remap_name][1] - real_obj_statistics[remap_name][0])
            
            if score_norm < min_score:
                min_score = score_norm
                min_class = remap_name
        samples.append([x['env'], x['episode'], x['step'], min_score, min_class])

# 排序
sorted_samples = sorted(samples, key=lambda x: x[3])  # 根据 min_score 升序排序
with open(f"./samples_{note}.pkl", "wb") as f:
    pickle.dump(sorted_samples, f)
scores = [x[3] for x in sorted_samples]


# load
# with open(f"./samples_{note}.pkl", "rb") as f:
#     sorted_samples = pickle.load(f)
# scores = [x[3] for x in sorted_samples]


print(f"max {max(scores)}, min {min(scores)}")
# 统计
print(f"number of samples :{len(scores)}")
# 设置直方图的区间大小
bin_size = 0.1
bins = np.arange(min(scores), max(scores) + bin_size, bin_size)
# 创建画布和主坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制样本数直方图（左纵轴）
counts, _, patches = ax1.hist(scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7,)
ax1.set_xlabel('score', fontsize=12)
ax1.set_ylabel('sample number', fontsize=12)
ax1.set_title('hist(num vs percentage)', fontsize=14)
# 在每个柱子上显示样本数
ax1.bar_label(patches, fmt='%d', label_type='edge')  # 显示整数样本数

# 创建次坐标轴（右纵轴）
ax2 = ax1.twinx()

# 绘制百分比直方图（右纵轴）
weights = np.ones_like(scores) / len(scores)  # 计算百分比
ax2.hist(scores, bins=bins, weights=weights)
ax2.set_ylabel('percentage', fontsize=12)

# 显示网格和调整布局
ax1.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig(f"./score_hist_{note}.png")