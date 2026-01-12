import os
import cv2
from vqf_utils import return_bin_idx
import clip
from PIL import Image
import torch

refr = "refri"

datas_vsqf = {
    "0_1_eqpmt": [2.44, 320],
    "0_2_eqpmt": [2.66, None],
    "0_4_eqpmt": [2.01, 180],
    "0": [1.19, 315],
    "1_0": [1.89, 40],
    "2_0": [2.86, 315],
    "3_0": [1.67, 315],
    "3_1": [2.18, 315],
    "4_0": [2.63, 60],
    "5_0": [2.82, 60],
    "6_0": [1.91, 225],
    "7_0": [1.12, 185],
    "10_0_eqpmt": [2.23, 315],
    "10_0": [1.38, 180],
    "10_1": [1.37, 180],
    "10_2_eqpmt": [2, None],
    "10_3_eqpmt": [2.01, 315],
    "11_0_eqpmt": [2, 330],
    "11_0": [1.61, 180],
    "11_2_eqpmt": [2.0, 330],
    "11_4_eqpmt": [2, None],
    "11_6_eqpmt": [2.01, 0],
    "12_0_eqpmt": [2.0, 340],
    "12_0": [1.85, 180],
    "12_1_eqpmt": [2.02, 180],
    "12_1":[1.83, 0],
    "12_2_eqpmt": [2.01, 0],
    "12_3_eqpmt": [2, 0],
    "12_3": [1.81, 190],
    "12_4_eqpmt": [2.0, 340],
    "12_5_eqpmt": [2.01,0],
    "13_0_eqpmt": [1.31, 60],
    "13_1_eqpmt": [2.17, 315],
    "13_3_eqpmt": [1.38, 120],
    "13_5_eqpmt": [2.16, 300],
    "14_0_eqpmt": [2, 315],
    "15_0": [2.09, 270],
    "15_1": [3.23, 30],
    "16_1": [0.81, 45],
    "17_0": [1.78, 135],
    "18_0": [1.29, 300],
    "19_3_eqpmt": [1.09, None],
    
}
datas_vsqf2 = {
    "0_0_"+refr : [1.5, 300],
    "1_0_"+refr : [2.16, 315],
    "2_0_"+refr: [1.94, 330],
    "3_0_"+refr:[1.38, 350],
    "4_1_"+refr:[1.07, 45],
    "5_0_"+refr:[0.72, 60],
    "6_0_"+refr:[3.44, 315],
    "0_1": [0.58, 315],
    "7_0": [1.02, 60],
    "7_2": [2.40, 225],
    "8_0": [1.14, 100],
    "8_1": [1.67, 270],
    "9_0": [1.08, 180],
    "10_0": [0.93, 270],
    "10_1": [3.21, 260],
    "10_2": [2.04, 135],
    "10_3": [3.84, 135],
    "10_4": [3.09, 90],
    "10_5": [5.52, 90],
    "10_6": [4.05, 180],
    "10_7": [3.13, 90],
    "10_10": [2.37, 90],
    "11_2": [0.87, 230],
    "13_0":[3.25, 140],
    "13_2": [4.21, 100],
    "13_3": [4.01, 90],
    "14_0":[1.98, 125],
    "15_0": [2.02, 125],
    "16_0":[2.42, 80],
    "16_1": [5.51, 135],
    "16_2": [3.63, 130],
    "16_4": [4.77, 90],
    "16_5": [3.63, 100],
    "16_6": [8.31, 90],
    "17_0": [2.24, 270],
    "17_2": [5.47, 140],
    "18_0": [3.92, 100],
    "18_1": [5.53, 90],
    "18_2": [2.75, 100],
    "18_4": [2.22, 80],
    "19_0_eqpmt": [1.84, 315],
    "20_0_eqpmt": [1.43, 330],
    "21_0_eqpmt": [1.55, 0]
    
    
}
datas_vsqf3= {
    "1_0_eqpmt2": [0.24, 0],
    "2_0_eqpmt2": [0.51, 330],
    '3_2_eqpmt2': [1.22, 0],
    '4_0_eqpmt2': [1.78, 280,],
    '5_0_eqpmt2': [4.42, 280],
    '6_0_eqpmt2': [3.85, 315],
    '7_2_eqpmt2': [1.68, 250],
    '8_4_eqpmt2': [1.41, 200],
    '9_0_eqpmt2': [2.81, 200],
    
}
# CLIP打分
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
text_str = ["water dispenser."]  #'cabinet.']        # 'cabinet']# 
text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)


img_lst = ['19_0_eqpmt', '20_0_eqpmt', '21_0_eqpmt']
# ['10_3', '10_2', '11_2', '13_0', '14_0', '15_0', '16_2', '16_1', '18_0', '17_2', '16_5']
# ['10_1', '10_0', '16_0', '17_0', '18_4']
# ['16_1']
# ['8_0', '7_0', '9_0']
# ['1_0_'+refr, '2_0_'+refr, '3_0_'+refr, '4_1_'+refr, '5_0_'+refr, '6_0_'+refr, '0_0_'+refr]
# ['14_0_eqpmt']
# ['0', '1_0', '3_0', '5_0', '6_0', '7_0', '11_0', '10_0', '12_0', '12_1_eqpmt', '13_0_eqpmt', '12_3']
# ['1_0_eqpmt2', '2_0_eqpmt2']
# ['3_2_eqpmt2', '4_0_eqpmt2', '5_0_eqpmt2', '6_0_eqpmt2',]
# ['7_2_eqpmt2', '8_4_eqpmt2', '9_0_eqpmt2']
# ['2_0', '3_1', '18_0']
# ['12_5_eqpmt', '12_2_eqpmt', '13_1_eqpmt']
# ['0_1_eqpmt', '10_3_eqpmt', '11_6_eqpmt', '12_1', '12_3_eqpmt', '13_5_eqpmt']
dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/outputs_vsqf_2"
for img in img_lst:
    
    data = datas_vsqf2[img]
    depth_mean = data[0]
    azimuth = data[1]
    
    # 计算坐标索引
    distances = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    dis_idx = return_bin_idx(depth_mean, distances, 0.25)
    angles = [i for i in range(360)]
    angle_interval = 10
    angle_bins  =angles[::angle_interval]
    azimuth_idx = return_bin_idx(azimuth, angle_bins, angle_interval/ 2)

    img_pth = dir + "/rgb_obj_" + img + ".png"
    pil_img = Image.open(img_pth)
    image = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        score = logits_per_text.item()
    
    print(f"img {img}, depth, azimuth: ({dis_idx}, {azimuth_idx}), score {score}")

# 重计算深度
# pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/outputs_vsqf_2"
# for img in os.listdir(pth):
#     if "depth" in img and "png" in img:
#         depth_pth = pth + "/" +img
#         depth = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED)
#         mask_ = depth > 0
#         depth_mean = depth[mask_].sum() / mask_.sum()
#         print(f"img {img} , depth mean is {depth_mean}")
#         pass
    