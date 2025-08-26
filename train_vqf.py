import torch
import clip
import cv2
from PIL import Image
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random
from tqdm import *
from datetime import datetime
from src.finetune.dataset_utils import get_loader, SampleLoader

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



    
    
def load_dataset(dir, gt_mode, angle_interval, preprocess, train_augment):
    
    
    train_dataset = None 
    test_unseen_dataset = None
    test_seen_dataset = None
    
    classes = ['refrigerator', ]    #'toilet', 'bed', 'couch' ,'chair']
    datas = []
    cls_objects = {}
    
    # cnt = 0
    for cls_ in classes:
        cls_objects[cls_] = {}
        for obj_dir in os.listdir(dir+"/"+cls_):
            cls_objects[cls_][obj_dir] = []
            for distance in os.listdir(dir + "/"+cls_+"/"+obj_dir):
                for azimuth in range(360): 
                    # img_pth = dir + "/"+ obj_dir+"/"+distance+f"/render_{azimuth:03d}.png"                    
                    datas.append((cls_, obj_dir, distance, azimuth))
        #     cnt += 1
        #     if cnt > 2:
        #         break
        # cnt = 0
    objects_all = {}
    for cls_ in classes:
        objects_all[cls_] = list(cls_objects[cls_].keys())
    
    train_objs = {}
    test_unseen_objs = {}
    for cls_ in classes:
        train_objs[cls_] = objects_all[cls_][:int(len(objects_all[cls_]) * 0.8)]  # 每物体划分4/1 seen/unseen object
        test_unseen_objs[cls_] = objects_all[cls_][int(len(objects_all[cls_]) * 0.8):]
    
    train_obj_num = 0
    test_unseen_num = 0
    for cls_ in classes:
        train_obj_num += len(train_objs[cls_])
        test_unseen_num += len(test_unseen_objs[cls_])
    # print(f"train_obj_num: {train_obj_num}, test_unseen_num: {test_unseen_num}")
    
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
    values_gt = {}
    for cls_ in classes:
        gt_file = dir+"/"+cls_+"_clip.pkl"
        with open(gt_file, "rb") as f:
            values_gt[cls_] = pickle.load(f)     # values[obj_dir][distance], idnex-angle
    
    TrainDataset = AzimuthDataset(train_dataset, dir, values_gt, gt_mode,
                                  obj_num=train_obj_num, 
                                  interval=angle_interval, 
                                  preprocess=preprocess,
                                  augment=True,
                                  augment_preprocess=train_augment)
    TestSeenDataset = AzimuthDataset(test_seen_dataset, dir, values_gt, gt_mode, obj_num=train_obj_num, interval=angle_interval, preprocess=preprocess)
    TestUnseenDataset = AzimuthDataset(test_unseen_dataset, dir, values_gt, gt_mode, obj_num=test_unseen_num, interval=angle_interval, preprocess=preprocess)
    return TrainDataset, TestSeenDataset, TestUnseenDataset

from src.vqf_constants import category_maps

class RealAzimuthDataset(Dataset):
    def __init__(self, dir, scenes, gt_mode):
        
        self.gt_mode = gt_mode
        
        self.data_lst = []
        for scene in scenes:
            with open(dir+f"/{scene}_objects_value_clip.pkl", 'rb') as f:
                object_label_clip = pickle.load(f)
                    
            with open(dir+f"/{scene}_objects_index.pkl", "rb") as f:
                object_index = pickle.load(f)
                
            for key, value in object_index.items():
                env, epi = key
                for cat, cat_obj in value.items():
                    if len(cat_obj)> 0:
                        for scene_obj_id, v_ in cat_obj.items():
                            _, obj_score_arr = object_label_clip[(env, epi)][cat][scene_obj_id]
                            for step, index in v_.items():
                                dis_idx, azimuth_idx = object_index[(env, epi)][cat][scene_obj_id][step]
                            # data_lst[cat].append((env, epi, step, obj_score_arr))
                                self.data_lst.append((scene, env, epi, step, cat, scene_obj_id, obj_score_arr, dis_idx, azimuth_idx))
        
        self.samplers = {scene: SampleLoader(dir + "/data/"+scene) for scene in scenes}

        self.norm_scale = {
            'refrigerator': [ 13.9, 28.9,]
        }
        
    def __getitem__(self, idx):
        '''
            return 
                y: 归一化的clip分数, 11*(360 / interval)
                mask_: 矩阵有值的地方
        '''
        scene, env, epi, step, cat, scene_obj_id, obj_score_arr, dis_idx, azimuth_idx = self.data_lst[idx]  # obj_score_arr 未归一化
        obj_score_arr = obj_score_arr.copy()
        # 加载对应data和rgb
        sampler = self.samplers[scene]
        sample_data = sampler.get_sample_multimodality(env, epi, step, ["bbsgt", "rgb", "depth", "semantic"])
        rgb = sample_data['rgb'].data
        semantic = sample_data['semantic'].data
        mask_ = semantic == scene_obj_id
        
        rgb_obj = rgb * mask_[:,:,None]
        # 不centerize

        class_name = category_maps[cat]
        mask_ = obj_score_arr < 1e-5
        y = (obj_score_arr[mask_] - self.norm_scale[class_name][0]) / (self.norm_scale[class_name][1] - self.norm_scale[class_name][0])
        if self.gt_mode == "inconsistent":
            y = torch.concat([y[:, azimuth_idx:], y[:, :azimuth_idx]], dim=-1)
            mask_ = torch.concat([mask_[:, azimuth_idx:], mask_[:, :azimuth_idx]], dim=-1)
        return rgb_obj, y, mask_
    
    def __len__(self):
        return len(self.data_lst)

def helper_collate(batch):
    # x = torch.concat([T.ToTensor()(b[0]).unsqueeze(0) for b in batch], dim=0)
    x = torch.concat([b[0].unsqueeze(0) for b in batch], dim=0)
    y = torch.concat([b[1].unsqueeze(0) for b in batch], dim=0)
    azimuths = [b[2] for b in batch]
    return x, y, azimuths

class AzimuthDataset(Dataset):
    def __init__(
        self,
        data_lst,
        dir,
        gt_values_data,
        gt_mode, 
        obj_num,
        interval,
        preprocess,
        augment=False,
        augment_preprocess=None,
    ):
        self.data_dir = dir
        self.datas = data_lst 
        
        self.gt_dis = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
        self.gt_values_data = gt_values_data
        self.gt_mode = gt_mode
        assert self.gt_mode in ["consistent", "inconsistent"], "Invalid gt mode"
        self.norm_scale = {
            'refrigerator': [12.7, 29.61],
            'couch': [13.1, 28.85],
            'toilet': [14, 26.83],
            'chair': [14.6, 30.18],
            'bed': [11.9, 28.46]
        }
        
        self.obj_num = obj_num
        self.interval = interval
        self.preprocess = preprocess
        self.augment = augment
        self.augment_preprocess = augment_preprocess
        
        
    def __getitem__(self, idx):
        '''
        return:
            x: 经过预处理后
            y: len(gt_dis) * (360 / obj_interval), tensor, 归一化值
        '''
        cls_, obj_dir, distance, azimuth = self.datas[idx]
        
        img_pth = self.data_dir + "/" + cls_ + "/"+ obj_dir+"/"+distance+f"/render_{azimuth:03d}.png"
        x = Image.open(img_pth)
        try:
            x = self.preprocess(x)
        except:
            print(f"error in {img_pth}")    
        # if self.augment:
        #     assert self.augment_preprocess != None, "augmentation is none"
        #     y_trans = []
        #     if "crop" in self.augment_preprocess:
        #         trans = T.Compose([transforms['crop']])
        #         x = trans(x)
        #     if "hflip" in self.augment_preprocess:
        #         trans = transforms['hflip'](mode=self.gt_mode)
        #         x = trans.augment_image(x)
        #         y_trans.append(trans.augment_y)
        # gt: 0-360, 2-5.5m矩阵
        y = []
        obj_value_dict = self.gt_values_data[cls_][obj_dir]
        for d in self.gt_dis:
            y.append(obj_value_dict[d][::self.interval])
            
        y= torch.tensor(y, dtype=torch.float)
        
        max_ = max(self.norm_scale[cls_])
        min_ = min(self.norm_scale[cls_])
        y = (y - min_) / (max_ - min_)
        y = torch.ones_like(y) - y
        
        # if self.gt_mode == "inconsistent":
        #     y = torch.concat([y[:, azimuth:], y[:, :azimuth]], dim=-1)
            
        # if self.augment:
        #     for yt in y_trans:
        #         y = yt(y, azimuth)
        return x, y, azimuth
    
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
        return self.preprocess(x).to(self.device)
    
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
    # parameters
    lr = 1e-3
    epochs = 10
    batch_size = 128
    
    ### model
    ## clip
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    in_dim = 768*2
    model = VQFModel(device, in_dim, angle_interval)
    
    ### load dataset
    gt_mode = "consistent"
    ## augmentation in train
    train_augment = ['crop', 'hflip']
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
    test_real_dataloader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False, collate_fn=helper_collate)
    ### log

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    note=  ""
    log_dir = f"./vqf_logs/{timestamp}_{note}"
    os.makedirs(log_dir, exist_ok=True)
    

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
        train_loss = []
        for i, batch in enumerate(tqdm(train_dataloader)):
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
        save_dir = log_dir + "/images/predict_seen"
        cnt = 0
        test_seen_loss = []
        for i, batch in tqdm(enumerate(test_seen_dataloader)):
            
            
            if cnt % int(720 / batch_size) == 0:       # 每个物体有8×360×1/4 = 720张test， batch=16; 45轮后换物体
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
                visualize(predict[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                visualize(y[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            cnt += 1
            
        test_seen_loss = sum(test_seen_loss) / len(test_seen_loss)
        test_seen_losses.append(test_seen_loss)
        print(f"epoch {e}, test seen loss: ", test_seen_loss)
        
        if test_seen_loss < best_seen_loss:
            best_seen_loss = test_seen_loss
            torch.save(model.state_dict(), log_dir + "/best_seen_model.pth")


        save_dir = log_dir + "/images/predict_unseen"
        cnt = 0
        test_unseen_loss = []
        for i, batch in enumerate(test_unseen_dataloader):
            
            
            if cnt % int(2880 / batch_size) == 0:  # 每物体2880张，共11520 test , 4物体，batch=16; 
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
                visualize(predict[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                visualize(y[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            cnt += 1
        test_unseen_loss = sum(test_unseen_loss) / len(test_unseen_loss)
        test_unseen_losses.append(test_unseen_loss)
        print(f"epoch {e}, test seen loss: ", test_unseen_loss)
        
        # save
        if test_unseen_loss < best_loss:
            best_loss = test_unseen_loss
            torch.save(model.state_dict(), log_dir + "/best_unseen_model.pth")

        save_dir = log_dir + "/images/predict_real"
        cnt = 0
        test_real_loss = []
        for i, batch in enumerate(test_real_dataloader):
            
            # if cnt % int(2880 / batch_size) == 0:  
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            y = y[:, 3:, :]     # 从第3行开始
            mask = mask[:, 3:, :]
            
            with torch.no_grad():
                predict = model(x)

            y, predict = y.unsqueeze(1), predict.unsqueeze(1)
            test_loss = (coef_dssim * dssim_criterion(y*mask, predict*mask)  
                + coef_mae * model.mean_absolute_error_loss(y*mask, predict*mask) 
                + coef_si * si_criterion(y*mask, predict*mask))
            test_real_loss.append(test_loss.item())
            
            # visualize
            for j in range(len(y.shape[0])):
                if cnt % 5 == 0:
                    visualize(predict[j,:,:], "", save_dir, note=f"pred_"+str(j)+f"_{str(i)}_"+f"_e{e}")
                    visualize(y[j,:,:], "", save_dir, note=f"gt_"+str(j)+f"_{str(i)}_"+f"_e{e}")
            cnt += 1
        test_real_loss = sum(test_real_loss) / len(test_real_loss)
        test_real_losses.append(test_real_loss)
        print(f"epoch {e}, test real loss: ", test_real_loss)
        
        # save
        if test_real_loss < best_real_loss:
            best_real_loss = test_real_loss
            torch.save(model.state_dict(), log_dir + "/best_real_model.pth")
            
    torch.save(model.state_dict(), log_dir + "/last_model.pth")
    
    print("train loss: ", train_losses)
    print("test seen loss: ", test_seen_losses)
    print("test unseen loss: ", test_unseen_losses)
    
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
    
    

        

            
        