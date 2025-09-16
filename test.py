from paths import *
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image

import torch.nn.functional as F
from utils import *
from inference import *
import time
from huggingface_hub import hf_hub_download
import cv2
ckpt_path = "./models/largeEx2/dino_weight.pt"
# hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)
print(ckpt_path)

save_path = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('weight loaded')
val_preprocess   = AutoImageProcessor.from_pretrained("/home/users/wpp/Orient-Anything/models/dino/", local_files_only=True)
# val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')

# image_path = [
#     f'/home/users/wpp/Orient-Anything/images/3.5m-radius-640/render_{i:03d}.png'
#     for i in range(360)
#     # "/home/users/wpp/Orient-Anything/images/top/render.png",
#     # "/home/users/wpp/Orient-Anything/images/3m/render_050.png",
#     # "/home/users/wpp/Orient-Anything/images/3.5m/render_082.png",
#     # "/home/users/wpp/Orient-Anything/images/3.5m/render_082.png",
#     # '/home/users/wpp/CLIP/diagonal_angle_table.png',
# ]
idxs = [
    # 163, 204, 217, 220, 224, 238, 374, 548, 1252, 
#        1247, 1249, 1268, 1321,  1367, 1376, 1378, 1380, 1385, 1386, 1398, 1400, 1404, 1405, 1408, 1409, 1410, 
# 1411, 1412, 1423, 1424, 1427, 1428, 1429, 1431, 1433, 1434, 1435, 1445, 1446, 1447, 1449, 
# 1455, 1456, 1460, 1461, 1462, 1465, 1466, 1467, 1469, 1478, 1484, 1486, 1487, 1493, 1495, 1500, 
    # 1502, 1504, 1509, 1511, 1513, 1518, 1520, 1523, 1529, 1531, 1532, 1534, 1540, 1541, 1547, 
    # 1550, 1552, 1553, 1558, 1560, 1561, 1564, 1565, 1567, 1573, 1578, 1579, 1580, 1581, 1587, 
    # 1589, 1593, 1596, 
    # 2665, 2667, 2668, 2672, 2673, 2674, 2675, 2676, 2677, 2679, 2687, 2689, 2690, 2691, 2692, 2698, 
    # 2705, 2708, 2711, 2714, 2715, 2718, 2720, 2722, 2724, 2727, 2731, 2733, 2735, 2736, 2737, 2741, 
    # 2747, 2748, 2749, 2750, 2751, 2752, 2759, 2760, 2761, 2762, 2769, 2772, 2775, 2776, 2782, 2783, 
    # 2785, 2786, 2791, 2792, 2793, 2795, 2797, 2799, 2800, 2801, 2803, 2804, 2808, 2813, 
    # 2843, 2847, 2852, 2858, 2860, 2866, 2871, 2873, 2876, 2878, 2879, 2884, 2887, 2889, 2891, 2896,
    # 2900, 2907, 2911
    87, 95, 96, 108, 131, 132, 163, 168, 171, 174, 187, 189, 190, 193, 198, 199, 216, 217, 218, 
    227, 228, 230, 231, 234, 235, 244, 247, 248, 249, 251, 260, 264, 265, 268, 278, 283, 288, 290, 
    292, 293, 294, 296, 299, 300, 304, 307, 309, 310, 316, 320, 322, 325, 330, 331, 332, 339, 346
     
]

# [
    # 

image_path = [
# f"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/multiSens_test_val5/imgs_mask/mask_0_epi0_env2_step{str(i)}.png"
#  for i in idxs 
"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env1/masked_obj/epi1_env0_step498_num2_cls56.png"
]

grid_count = 9      #  保存名字
text_str = ["bed"]

time_all = 0.

rows = 2
cols = 5
images_per_grid = rows * cols

# 存储当前批次的图像和分数
current_images = []
current_scores = []
current_azimuths = []

margin = 60  # 底部留白
border_width = 3  # 边框宽度
border_color = (200, 200, 200)  # 浅灰色边框
font_scale = 0.8  # 字体大小
font_thickness = 2  # 字体粗细
font_color = (0, 0, 0)  # 黑色字体



for i, img_pth in enumerate(image_path):
    cv_img = cv2.imread(img_pth)
    
    origin_image = Image.open(img_pth).convert('RGB')
    start_t = time.time()
    angles = get_3angle(origin_image, dino, val_preprocess, device)
    end_t = time.time()
    time_all += (end_t - start_t)
    
    azimuth     = float(angles[0])
    polar       = float(angles[1])
    rotation    = float(angles[2])
    confidence  = float(angles[3])

    print(f' azimuth: {azimuth}, polar: {polar}, rotation: {rotation}, confidence: {confidence}')

    current_images.append(cv_img)
    current_scores.append(confidence)
    current_azimuths.append(azimuth)
    
    if len(current_images) == images_per_grid or i == len(image_path) - 1:
        # 计算实际需要处理的图像数量
        num_images = len(current_images)
        
        # 计算需要的行数（可能不足10张）
        actual_rows = 2 if num_images > cols else 1
        actual_cols = min(cols, num_images)

        # 获取单张图像尺寸
        img_height, img_width = current_images[0].shape[:2]
        
        # 创建大图（留出空间写分数）
        grid_width = actual_cols * img_width
        grid_height = actual_rows * (img_height + margin)
        grid_img = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)
        
         # 将图像粘贴到大图上并添加边框
        for idx in range(num_images):
            row = idx // actual_cols
            col = idx % actual_cols
            x_offset = col * img_width
            y_offset = row * (img_height + margin)
            
            # 绘制边框
            cv2.rectangle(grid_img, 
                         (x_offset, y_offset),
                         (x_offset + img_width - 1, y_offset + img_height - 1),
                         border_color, border_width)
            
            # 粘贴图像（向内缩进边框宽度）
            grid_img[y_offset+border_width:y_offset+img_height-border_width, 
                    x_offset+border_width:x_offset+img_width-border_width] = current_images[idx][border_width:-border_width, border_width:-border_width]
            
            # 添加分数标注
            score_text = f"{image_path[i - num_images + 1 + idx].split('/')[-1][:-4]}" 
            score_text2 = f"{current_azimuths[idx]:.3f} :{current_scores[idx]:.2f} "
            # 计算文本位置（居中）
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_offset + (img_width - text_size[0]) // 2
            text_y = y_offset + img_height + 15
            
            text_size2 = cv2.getTextSize(score_text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x2 = x_offset + (img_width - text_size[0]) // 2
            text_y2 = y_offset + img_height + 40
            
            # 绘制文本
            cv2.putText(grid_img, score_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            cv2.putText(grid_img, score_text2, (text_x2, text_y2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        
        # 保存网格图
        # cv2.imwrite(f"./images/gibson/{text_str[0]}/OriAny_{grid_count}.png", grid_img)
        cv2.imwrite(f"./images/gibson/{text_str[0]}/{image_path[i - num_images + 1 + idx].split('/')[-1]}", grid_img)
        grid_count += 1
        
        # 重置当前批次
        current_images = []
        current_scores = []
        
print(f'Inference time: {time_all:.2f} seconds')



