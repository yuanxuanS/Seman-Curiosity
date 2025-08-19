import torch
import clip
import cv2
from PIL import Image
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
import matplotlib.pyplot as plt

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features1: Optional[int] = None,
        hidden_features2: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        final_bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features1 = hidden_features1 or in_features
        self.fc1 = nn.Linear(in_features, hidden_features1, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features1, hidden_features2, bias=bias)
        self.fc3 = nn.Linear(hidden_features2, out_features, bias=final_bias)
        self.drop = nn.Dropout(drop)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        
        x = self.fc2(x)
        x = self.act(x)
        # x = self.drop(x)
        
        x = self.fc3(x)
        # x = self.drop(x)        # activation?
        x = self.sigmoid(x)
        return x

from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random


    
    
def load_dataset(dir, gt_file, angle_interval, preprocess):
    
    
    train_dataset = None 
    test_unseen_dataset = None
    test_seen_dataset = None
    
    classes = ['refrigerator', 'chair',]        #  'couch', 'toilet', 'bed']
    datas = []
    cls_objects = {}
    
    for cls_ in classes:
        cls_objects[cls_] = {}
        for obj_dir in os.listdir(dir+"/"+cls_):
            cls_objects[cls_][obj_dir] = []
            for distance in os.listdir(dir + "/"+cls_+"/"+obj_dir):
                for azimuth in range(360): 
                    # img_pth = dir + "/"+ obj_dir+"/"+distance+f"/render_{azimuth:03d}.png"
                    
                    datas.append((cls_, obj_dir, distance, azimuth))
                    # img_ = Image.open(img_pth)
                    # images.append(img_)
    objects_all = {}
    for cls_ in classes:
        objects_all[cls_] = list(cls_objects[cls_].keys())
    
    train_objs = {}
    test_unseen_objs = {}
    for cls_ in classes:
        train_objs[cls_] = objects_all[cls_][:int(len(objects_all[cls_]) * 0.8)]  # 每物体划分4/1 seen/unseen object
        test_unseen_objs[cls_] = objects_all[cls_][int(len(objects_all[cls_]) * 0.8):]
    
    train_dataset = []     #  该类别物体, 每个物体多个角度划分 3/1 
    test_seen_dataset = []
    for cls_ in classes:
        for obj_dir in train_objs[cls_]:
            all_data = []
            for distance in os.listdir(dir + "/"+cls_+"/" + obj_dir):
                random_idxs = [idx for idx in range(360)]
                random.shuffle(random_idxs)
                all_data_ = [(cls_, obj_dir, distance, azimuth) for azimuth in random_idxs]
                all_data.extend(all_data_)
            train_dataset.extend(all_data[:int(len(all_data)*0.75)])
            test_seen_dataset.extend(all_data[int(len(all_data)*0.75):])
        
    
    test_unseen_dataset = []
    for cls_ in classes:
        for obj_dir in test_unseen_objs[cls_]:
            for distance in os.listdir(dir + "/"+cls_+"/" +obj_dir):
                for azimuth in range(360): 
                    test_unseen_dataset.append((cls_, obj_dir, distance, azimuth))

    # gt
    with open(gt_file, "rb") as f:
        values_gt = pickle.load(f)     # values[obj_dir][distance], idnex-angle
    
    TrainDataset = AzimuthDataset(train_dataset, dir, values_gt, interval=angle_interval, preprocess=preprocess)
    TestSeenDataset = AzimuthDataset(test_seen_dataset, dir, values_gt, interval=angle_interval, preprocess=preprocess)
    TestUnseenDataset = AzimuthDataset(test_unseen_dataset, dir, values_gt, interval=angle_interval, preprocess=preprocess)
    return TrainDataset, TestSeenDataset, TestUnseenDataset
    
class AzimuthDataset(Dataset):
    def __init__(
        self,
        data_lst,
        dir,
        gt_values_data,
        interval,
        preprocess,
    ):
        self.data_dir = dir
        self.datas = data_lst 
        
        self.gt_dis = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
        self.gt_values_data = gt_values_data
        self.norm_scale = {
            'refrigerator': [28.9, 13.9]
        }
        
        self.interval = interval
        self.preprocess = preprocess
        
        
    def __getitem__(self, idx):
        cls_, obj_dir, distance, azimuth = self.datas[idx]
        
        img_pth = self.data_dir + "/" + cls_ + "/"+ obj_dir+"/"+distance+f"/render_{azimuth:03d}.png"
        x = Image.open(img_pth)
        x = self.preprocess(x)
        # gt: 0-360, 2-5.5m矩阵
        y = []
        obj_value_dict = self.gt_values_data[obj_dir]
        for d in self.gt_dis:
            y.append(obj_value_dict[d][::self.interval])
            
        y= torch.tensor(y, dtype=torch.float)
        
        max_ = max(self.norm_scale['refrigerator'])
        min_ = min(self.norm_scale['refrigerator'])
        y = (y - min_) / (max_ - min_)
        y = torch.ones_like(y) - y
        return x, y
    
    def __len__(self):
        return len(self.datas)
    
class VQFModel(nn.Module):
    def __init__(self, device, in_dims, angle_interval, ):
        super().__init__()
        
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device) 
        ## dinov2
        self.dinov2 = torch.hub.load('/home/users/wpp/dinov2', 'dinov2_vitb14',source='local').to(device)  # base; 16patch

        self.angle_bin = int(360 / angle_interval)
        out_dims = self.angle_bin * 8
        self.mlp = Mlp(in_dims, 
                  int(in_dims / 2),
                  int(in_dims / 4),
                  out_dims,
                  ).to(device)
    def preprocess(self, x):
        return self.preprocess(x).unsqueeze(0).to(self.device)
    
    def forward(self, x):
        with torch.no_grad():
            patch_embedding = self.clip_model.image_patch_embedding(x)
            embedding_clip = torch.mean(patch_embedding, dim=1)
            # embedding_clip = torch.max(patch_embedding, dim=1)

            ## dino
            features_dict = self.dinov2.forward_features(x)
            patch_embedding_dino = features_dict['x_norm_patchtokens']
            embedding_dino = torch.mean(patch_embedding_dino, dim=1)
            
            # concat
            embedding = torch.concat([embedding_clip, embedding_dino], dim=-1)
        
        # mlp
        output = self.mlp(embedding)
        output = output.reshape(x.shape[0], -1, self.angle_bin)
        return output
    
    def mean_absolute_error_loss(self, x, target):
        # 
        criterion = nn.L1Loss(reduction='mean')
        loss = criterion(x, target)
        return loss

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
    ### model
    ## clip
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    in_dim = 768*2
    model = VQFModel(device, in_dim, angle_interval)
    
    ### load dataset
    class_name = "refrigerator"
    dir = '/data2/wpp_data/obj_azimuth/'
    gt_file = class_name+"_clip.pkl"
    train_dataset, test_seen_dataset, test_unseen_dataset = load_dataset(dir, gt_file, angle_interval, preprocess=model.preprocess)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=batch_size, shuffle=False)
    test_unseen_dataloader = DataLoader(test_unseen_dataset, batch_size=batch_size, shuffle=False)
    
    
    
    # parameters
    lr = 1e-3
    epochs = 10
    
    ### loss
    from dssim_loss import SSIMLoss, ScaleInvariantLoss
    coef_mae, coef_dssim, coef_si = 0.3, 0.4, 0.3
    dssim_criterion = SSIMLoss(5)
    si_criterion = ScaleInvariantLoss(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=0.001)
    
    # train
    train_losses = []
    for e in range(epochs):
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            predict = model(x)
            
            # loss
            y, predict = y.unsqueeze(1), predict.unsqueeze(1)
            loss = (coef_dssim * dssim_criterion(y, predict)  
                    + coef_mae * model.mean_absolute_error_loss(y, predict) 
                    + coef_si * si_criterion(y, predict))
            train_loss+= loss.item()
            print(f"loss is {loss.item()}")
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)
        print(f"epoch {e}, loss: ", train_losses)
    
        # test
        save_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/predict_seen"
        cnt = 0
        for i, batch in enumerate(test_seen_dataloader):
            
            
            if cnt % 45 == 0:       # 每个物体有720张test， batch=16; 45轮后换物体
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    predict = model(x)
                # for j in range(x.shape[0]):
                
                j = 0
                visualize(predict[j,:,:], class_name, save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                visualize(y[j,:,:], class_name, save_dir, note=f"gt_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            cnt += 1
            
        save_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/predict_unseen"
        cnt = 0
        for i, batch in enumerate(test_unseen_dataloader):
            
            
            if cnt % 180 == 0:  # 共11520 test , 4物体，batch=16; 720轮换物体
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    predict = model(x)
            # for j in range(x.shape[0]):
                j = 0
                visualize(predict[j,:,:], class_name, save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                visualize(y[j,:,:], class_name, save_dir, note=f"gt_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            cnt += 1
    # with torch.no_grad():
        
        # print(output.shape)
        
        
        

        

            
        