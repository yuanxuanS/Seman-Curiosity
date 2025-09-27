import torch
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import os
import random
from tqdm import *
from datetime import datetime

import numpy as np
from vqf import VQFModel
from vqf_dataset import load_dataset, RealAzimuthDataset, helper_collate

w, h = 640, 640
import torchvision.transforms as T
from torchvision.transforms import functional as TF

class HorizontalFlip:

    def __init__(self, mode="consistant") -> None:
        """HorizontalFlip, 水平翻转
        """
        super().__init__()
        self.p = random.uniform(0, 1)
        self.thres = 0.5
        self.mode = mode
    def augment_image(self, image):
        if self.p < self.thres:
            return TF.hflip(image)
        else:
            return image
    def augment_y(self, y, azimuth):
        # if self.p < self.thres:
        if self.mode == "consistent":
            return y
        elif self.mode == "inconsistent":  # y从当前角度开始，角度增加直到循环回来
            idx = -2*azimuth-1
            y = torch.concat([y[:, idx:], y[:, :idx]], dim=-1)
            return y    
   
   
transforms = {"crop": T.RandomResizedCrop((224,224), scale=(0.3, 1.0), ratio=(0.75, 1.33)) , # scale crop大小为原来的多少倍； ratio: 长宽比
              "hflip": HorizontalFlip}

   

def visualize(x, obj_str, save_dir, note=""):
    '''
    x : h*w
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    # 绘制热力图
    plt.figure(figsize=(18, 3))  # 宽度调大以适应360列
    im = plt.imshow(x, 
                    cmap='hot',  # 颜色映射
                    aspect='auto',   # 自动调整纵横比防止变形
                    interpolation='none')  # 禁用插值保持清晰边界
    plt.colorbar(im, label='Value')  # 添加颜色条
    plt.title('value Heatmap')
    plt.xlabel('azimuth (0-359)')
    plt.ylabel('distance')
    os.makedirs(save_dir+"/"+obj_str, exist_ok=True)
    plt.savefig(save_dir+"/"+obj_str+"/"+note+".png", dpi=300, bbox_inches='tight')  # 保存高清图
    plt.show()
    
if __name__ == "__main__":
    
    angle_interval = 10
    # parameters
    lr = 1e-3
    epochs = 10
    batch_size =1024
    withdrop = False
    drop=0.5
    load_model = True
    model_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/08-29_17-25-03_/best_real_model_e2.pth"
    # "./vqf_logs/08-28_16-53-57_/best_unseen_model_e1.pth"
    eval_only = True
    # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/08-27_09-48-23_/best_unseen_model.pth"
    mode =   "11dis" #"8dis"   #
    ### model
    ## clip
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    in_dim = 768*2
    distances = ['0.5m', '1m', '1.5m', '2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
    distances = distances if mode == "11dis" else distances[3:]
    
    ### log

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    note=  "eval"
    log_dir = f"./vqf_logs/{timestamp}_{note}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"log dir: {log_dir} in device {device}")
    
    model = VQFModel(device, in_dim, withdrop, drop, angle_interval, distance_len=len(distances))
    if load_model:
        model.load_state_dict(torch.load(model_pth))
    
    ### load dataset
    gt_mode = "consistent"
    ## augmentation in train
    train_augment = ['crop', 'hflip']       # 
    augments_x = []
    augments_y = []
    if "crop" in train_augment:
        trans = T.Compose([transforms['crop']])
        augments_x.append(trans)
        augments_y.append(None)
    if "hflip" in train_augment:
        trans_hflip = [transforms['hflip'](mode=gt_mode) for _ in range(10000)]
        hflips_x, hflips_y = [], []
        for _ in range(batch_size):
            trans = trans_hflip[random.randint(0, 10000-1)]
            hflips_x.append(trans.augment_image)
            hflips_y.append(trans.augment_y)
        augments_x.append(hflips_x)
        augments_y.append(hflips_y)
    
    # class_name = "refrigerator"
    dataset_dir = '/data2/wpp_data/obj_azimuth/'
    train_dataset, test_seen_dataset, test_unseen_dataset = load_dataset(dataset_dir, 
                                                                         gt_mode,
                                                                         angle_interval, 
                                                                         distances=distances,
                                                                         preprocess=model.preprocess,
                                                                         train_augment=train_augment)
    
    dir = "/data1/wpp_data/data/obj_samples/"
    scenes = ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]
    test_real_dataset = RealAzimuthDataset(dir, scenes, gt_mode)
    
    test_seen_num = test_seen_dataset.obj_num
    test_unseen_num = test_unseen_dataset.obj_num
    
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=helper_collate)
    test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=batch_size, shuffle=False, collate_fn=helper_collate)
    test_unseen_dataloader = DataLoader(test_unseen_dataset, batch_size=batch_size, shuffle=False, collate_fn=helper_collate)
    test_real_dataloader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False)
    

    ### loss
    from dssim_loss import SSIMLoss, ScaleInvariantLoss
    coef_mae, coef_dssim, coef_si = 0.3, 0.4, 0.3
    dssim_criterion = SSIMLoss(5)
    si_criterion = ScaleInvariantLoss(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=0.001)
    
    # train
    train_losses, test_seen_losses, test_unseen_losses,test_real_losses  = [], [], [], []
    best_loss = 1e5     # seen
    best_seen_loss = 1e5
    best_real_loss = 1e5
    for e in range(epochs):
        
        if not eval_only:
            train_loss = []
            for i, batch in enumerate(tqdm(train_dataloader, desc="train:")):
                x, y, azimuth = batch
                
                # x = model.preprocess(x)
                x, y = x.to(device), y.to(device)
                # augment
                for aug_x, aug_y in zip(augments_x, augments_y):

                    if not isinstance(aug_x, list):
                        x = aug_x(x)
        
                    if aug_y != None:
                        if isinstance(aug_y, list):
                            for idx in range(len(batch)):
                                x[idx] = aug_x[idx](x[idx])
                                y[idx] = aug_y[idx](y[idx], azimuth[idx])
                
                # forward
                predict = model(x)
                
                # loss
                y, predict = y.unsqueeze(1), predict.unsqueeze(1)
                loss = (coef_dssim * dssim_criterion(y, predict)  
                        + coef_mae * model.mean_absolute_error_loss(y, predict) 
                        + coef_si * si_criterion(y, predict))
                train_loss.append(loss.item())
                # print(f"loss is {loss.item()}")
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_loss = sum(train_loss) / len(train_loss)
            train_losses.append(train_loss)
            print(f"epoch {e}, loss: ", train_loss)
        
    
        ###  test
        # ''''''
        save_dir = log_dir + "/images/predict_seen"
        cnt = 0
        test_seen_loss = []
        for i, batch in enumerate(tqdm(test_seen_dataloader, desc="test seen:")):
            
            
            # if cnt % int(720 / batch_size) == 0:       # 每个物体有8×360×1/4 = 720张test， batch=16; 45轮后换物体
            x, y, azimuth = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                predict = model(x)
            
            y, predict = y.unsqueeze(1), predict.unsqueeze(1)
            test_loss = (coef_dssim * dssim_criterion(y, predict)  
                + coef_mae * model.mean_absolute_error_loss(y, predict) 
                + coef_si * si_criterion(y, predict))
            test_seen_loss.append(test_loss.item())
                
                # visualize
            j = 0
            visualize(predict.squeeze(1)[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            if e == 0:
                visualize(y.squeeze(1)[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}")
            cnt += 1
            
        test_seen_loss = sum(test_seen_loss) / len(test_seen_loss)
        test_seen_losses.append(test_seen_loss)
        print(f"epoch {e}, test seen loss: ", test_seen_loss)
        
        if test_seen_loss < best_seen_loss:
            best_seen_loss = test_seen_loss
            torch.save(model.state_dict(), log_dir + f"/best_seen_model_e{e}.pth")

        # test on unseen dataset
        save_dir = log_dir + "/images/predict_unseen"
        cnt = 0
        test_unseen_loss = []
        for i, batch in enumerate(tqdm(test_unseen_dataloader, desc="test unseen:")):
            # if cnt % int(2880 / batch_size) == 0:  # 每物体2880张，共11520 test , 4物体，batch=16; 
            x, y, azimuth = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                predict = model(x)

            y, predict = y.unsqueeze(1), predict.unsqueeze(1)
            test_loss = (coef_dssim * dssim_criterion(y, predict)  
                + coef_mae * model.mean_absolute_error_loss(y, predict) 
                + coef_si * si_criterion(y, predict))
            test_unseen_loss.append(test_loss.item())
            
            # visualize
            j = 0
            visualize(predict.squeeze(1)[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            if e == 0:
                visualize(y.squeeze(1)[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}")
            cnt += 1
        test_unseen_loss = sum(test_unseen_loss) / len(test_unseen_loss)
        test_unseen_losses.append(test_unseen_loss)
        print(f"epoch {e}, test unseen loss: ", test_unseen_loss)
        
        # save
        if test_unseen_loss < best_loss:
            best_loss = test_unseen_loss
            torch.save(model.state_dict(), log_dir + f"/best_unseen_model_e{e}.pth")
        # '''
        
        # test on real dataset
        save_dir = log_dir + "/images/predict_real"
        cnt = 0
        test_real_loss = []
        for i, batch in enumerate(tqdm(test_real_dataloader, desc="test real: ")):
            
            # if cnt % int(2880 / batch_size) == 0:  
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            idx = 3 if mode == "8dis" else 0
            y = y[:, idx:, :]     # 从第3行开始
            mask = mask[:, idx:, :]
            
            with torch.no_grad():
                predict = model(x)

            y, predict = y.unsqueeze(1), predict.unsqueeze(1)
            mask = mask.unsqueeze(1)
            test_loss = (coef_dssim * dssim_criterion(y*mask, predict*mask)  
                + coef_mae * model.mean_absolute_error_loss(y*mask, predict*mask) 
                + coef_si * si_criterion(y*mask, predict*mask))
            test_real_loss.append(test_loss.item())
            
            # visualize
            for j in range(y.shape[0]):
                if cnt % 5 == 0:
                    visualize((predict*mask).squeeze(1)[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                    if e == 0:
                        visualize(y.squeeze(1)[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}")
            cnt += 1
        test_real_loss = sum(test_real_loss) / len(test_real_loss)
        test_real_losses.append(test_real_loss)
        print(f"epoch {e}, test real loss: ", test_real_loss)
        
        # save
        if test_real_loss < best_real_loss:
            best_real_loss = test_real_loss
            torch.save(model.state_dict(), log_dir + f"/best_real_model_e{e}.pth")
        
        print("train loss: ", train_losses)
        print("test seen loss: ", test_seen_losses)
        print("test unseen loss: ", test_unseen_losses)
    torch.save(model.state_dict(), log_dir + "/last_model.pth")
    
    print("final train loss: ", train_losses)
    print("final test seen loss: ", test_seen_losses)
    print("final test unseen loss: ", test_unseen_losses)
    
    # visualize loss
    plt.figure()  # 宽度调大以适应360列
    plt.plot(train_losses, label="train")
    plt.plot(test_seen_losses, label="test_seen")
    plt.plot(test_unseen_losses, label="test_unseen")
    plt.legend()
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(log_dir + "/loss.png", dpi=300, bbox_inches='tight')  # 保存高清图
    plt.show()
    
    

        

            
        