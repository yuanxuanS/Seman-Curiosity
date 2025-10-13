import torch
from datetime import datetime
import os
from vqf import VQFModel
from vqf_dataset import load_dataset, RealAzimuthDataset, helper_collate
from vqf_train import visualize
import numpy as np
import pickle
import logging
import cv2
import torchvision.transforms as T
from PIL import Image

if __name__ == "__main__":
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    in_dim = 768*2
    withdrop = False
    drop=0.5
    angle_interval = 10
    mode =   "11dis" #"8dis"   #
    distances = ['0.5m', '1m', '1.5m', '2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
    distances = distances if mode == "11dis" else distances[3:]
    
    ### log
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    note=  "eval_single_real_incosis"
    log_dir = f"./vqf_logs/{timestamp}_{note}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"log dir: {log_dir} in device {device}")
    
    log_name = 'eval.log'
    logging.basicConfig(
        filename=log_dir + log_name,
        level=logging.INFO)
    log = " "
    
    ### model
    model_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/10-05_23-30-23_incnsistnt/best_real_model_e0.pth"
    # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/08-29_17-25-03_/best_real_model_e2.pth"
    model = VQFModel(device, in_dim, withdrop, drop, angle_interval, distance_len=len(distances))
    model.load_state_dict(torch.load(model_pth))
    
    gt_mode = "inconsistent"
    dir = "/data1/wpp_data/data/obj_samples/"
    scenes = ["Corozal", "Darden", "Markleeville", "Wiconisco"]
    test_real_dataset = RealAzimuthDataset(dir, scenes, gt_mode)
    
    # scene, env, epi, step, cat, scene_obj_id = "Darden", 1, 0, 263, 0, 24
    # x, y, mask = test_real_dataset.get_single_data(scene, env, epi, step, cat, scene_obj_id)
    # x, y, mask = x, torch.from_numpy(y), torch.from_numpy(mask)
    # x, y, mask = x.to(device), y.to(device), mask.to(device)
    # x, y, mask = x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)
    # print(f"x {x.shape}, y {y.shape} mask {mask.shape}")
    
    # 读入多张图
    dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/outputs_vsqf_2"
    transforms = T.Compose([
            T.Resize(size=224, interpolation=Image.BICUBIC),
            T.CenterCrop(size=(224, 224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        ])
    for img in os.listdir(dir):
        if not "rgb" in img:
            continue
        pth = dir + "/"+img
        x = cv2.imread(pth)
        x = x.transpose((2,0,1))
        x = transforms(torch.from_numpy(x).type(torch.float32))
        x = x.to(device)
        x = x.unsqueeze(0)
        

    
        idx = 3 if mode == "8dis" else 0
        # y = y[:, idx:, :]     # 从第3行开始
        # mask = mask[:, idx:, :]
        
        with torch.no_grad():
            predict = model(x)
        
        # print(f"predict shape {predict.shape}, mask shape {mask.shape}")
        # predict = predict*mask
        save_dir = log_dir + "/images/predict_real"
        # visualize(predict[0,:,:], "", save_dir, note=f"input_scene{scene}_env{env}_epi{epi}_cat{cat}_objid{scene_obj_id}_step{step}")
        visualize(predict[0,:,:], "", save_dir, note=f"input_{img[:-4]}")
    
        # 选取最值
        # max_value = False
        K = 5
        predict = predict
        # 排序并获取前 K 个最大值的索引
        sorted_indices = np.argsort(predict.cpu().numpy().flatten())[-K:]
        row_indices, col_indices = np.unravel_index(sorted_indices, predict.squeeze(0).cpu().numpy().shape)   # 从0开始
        
        # 获取对应的值和索引
        top_k_values = predict.squeeze(0).cpu().numpy()[row_indices, col_indices]
        top_k_indices = list(zip(row_indices, col_indices))
        log += f"{img}"
        log += f"\n max sample value Top-{K} 值: {top_k_values}"
        log += f"\n max sample value Top-{K} 索引(dis, azimuth): {top_k_indices}"
        
        #  排序并获取前 K 个最小值的索引
        # predict[~mask] = 1e4
        sorted_indices = np.argsort(predict.cpu().numpy().flatten())[:K]
        row_indices, col_indices = np.unravel_index(sorted_indices, predict.squeeze(0).cpu().numpy().shape)   # 从0开始

        # 获取对应的值和索引
        top_k_values = predict.squeeze(0).cpu().numpy()[row_indices, col_indices]
        top_k_indices = list(zip(row_indices, col_indices))
        log += f"\n min sample value Top-{K} 值: {top_k_values}"
        log += f"\n min sample value Top-{K} 索引(dis, azimuth): {top_k_indices}"
        
        # 获取对应索引
        pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/obj_samples/"
        # with open(pth + f"{scene}_objects_index.pkl", "rb") as f:
        #     object_index = pickle.load(f)
        # obj_idx = object_index[(env, epi)][cat][scene_obj_id]
        # for step, info in obj_idx.items():
        #     dis_idx, azimuth_idx = info
        #     log += f"\n env{env} epi{epi} cat {cat} obj_id{scene_obj_id} step{step}: dis {dis_idx} azimuth {azimuth_idx}"
        logging.info(log)