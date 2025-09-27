from PIL import Image
import os
import pickle
import random
from torch.utils.data import Dataset
from src.finetune.dataset_utils import get_loader, SampleLoader
import torchvision.transforms as T
import torch
import numpy as np

def helper_collate(batch):
    # x = torch.concat([T.ToTensor()(b[0]).unsqueeze(0) for b in batch], dim=0)
    x = torch.concat([b[0].unsqueeze(0) for b in batch], dim=0)
    y = torch.concat([b[1].unsqueeze(0) for b in batch], dim=0)
    azimuths = [b[2] for b in batch]
    return x, y, azimuths

def load_dataset(dir, gt_mode, angle_interval, distances, preprocess, train_augment):
    
    
    train_dataset = None 
    test_unseen_dataset = None
    test_seen_dataset = None
    
    classes = ['refrigerator', 'toilet', 'bed', 'couch' ,'chair']
    # distances = ['0.5m', '1m', '1.5m', '2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
    datas = []
    cls_objects = {}
    
    # cnt = 0
    for cls_ in classes:
        cls_objects[cls_] = {}
        for obj_dir in os.listdir(dir+"/"+cls_):
            cls_objects[cls_][obj_dir] = []
            # for distance in os.listdir(dir + "/"+cls_+"/"+obj_dir):
            for distance in distances:
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
            # for distance in os.listdir(dir + "/"+cls_+"/" + obj_dir):
            for distance in distances:
                random_idxs = [idx for idx in range(360)]
                random.shuffle(random_idxs)
                all_data_ = [(cls_, obj_dir, distance, azimuth) for azimuth in random_idxs]
                all_data.extend(all_data_)
            train_dataset.extend(all_data[:int(len(all_data)*0.75)])
            test_seen_dataset.extend(all_data[int(len(all_data)*0.75):])
        
    
    test_unseen_dataset = []
    for cls_ in classes:
        for obj_dir in test_unseen_objs[cls_]:
            # for distance in os.listdir(dir + "/"+cls_+"/" +obj_dir):
            for distance in distances:
                for azimuth in range(360): 
                    test_unseen_dataset.append((cls_, obj_dir, distance, azimuth))

    # gt
    values_gt = {}
    for cls_ in classes:
        gt_file = dir+"/"+cls_+"_clip_add.pkl"
        with open(gt_file, "rb") as f:
            values_gt[cls_] = pickle.load(f)     # values[obj_dir][distance], idnex-angle
    
    TrainDataset = AzimuthDataset(train_dataset, dir, distances, values_gt, gt_mode,
                                  obj_num=train_obj_num, 
                                  interval=angle_interval, 
                                  preprocess=preprocess,
                                  augment=True,
                                  augment_preprocess=train_augment)
    TestSeenDataset = AzimuthDataset(test_seen_dataset, dir,distances, values_gt, gt_mode, obj_num=train_obj_num, interval=angle_interval, preprocess=preprocess)
    TestUnseenDataset = AzimuthDataset(test_unseen_dataset, dir, distances, values_gt, gt_mode, obj_num=test_unseen_num, interval=angle_interval, preprocess=preprocess)
    return TrainDataset, TestSeenDataset, TestUnseenDataset

from src.vqf_constants import category_maps

class RealAzimuthDataset(Dataset):
    def __init__(self, dir, scenes, gt_mode):
        
        self.gt_mode = gt_mode
        self.dir = dir
        
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
                            if obj_score_arr.max() > 0.1:
                                for step, index in v_.items():
                                    dis_idx, azimuth_idx = object_index[(env, epi)][cat][scene_obj_id][step]
                                # data_lst[cat].append((env, epi, step, obj_score_arr))
                                    self.data_lst.append((scene, env, epi, step, cat, scene_obj_id, obj_score_arr, dis_idx, azimuth_idx))
        
        self.samplers = {scene: SampleLoader(dir + "/data/"+scene) for scene in scenes}

        self.norm_scale = {     # real object 统计
            'chair': [15.9, 30.1],
            'couch':  [13.4, 30.4], 
            'bed': [14.9, 28.6],
            'toilet': [13.5, 27.3],
            'refrigerator': [ 13.2, 30.4,]
        }
        
        self.transforms = T.Compose([
            T.Resize(size=224, interpolation=Image.BICUBIC),
            T.CenterCrop(size=(224, 224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        ])
        
    def __getitem__(self, idx):
        '''
            return 
                y: 归一化的clip分数, 11*(360 / interval)
                mask_: 矩阵有值的地方
        '''
        scene, env, epi, step, cat, scene_obj_id, obj_score_arr, dis_idx, azimuth_idx = self.data_lst[idx]  # obj_score_arr 未归一化
        # obj_score_arr = obj_score_arr.copy()
        # 加载对应data和rgb
        sampler = self.samplers[scene]
        sample_data = sampler.get_sample_multimodality(env, epi, step, ["bbsgt", "rgb", "depth", "semantic"])
        rgb = sample_data['rgb'].data
        semantic = sample_data['semantic'].data
        mask_rgb = semantic == scene_obj_id
        
        rgb_obj = rgb * mask_rgb[:,:,None]
        rgb_obj = rgb_obj.transpose((2,0,1))
        rgb_obj = self.transforms(torch.from_numpy(rgb_obj).type(torch.float32))
        # 不centerize

        class_name = category_maps[cat]
        mask_ = obj_score_arr > 1e-5        # 0 处为无值
        y = obj_score_arr.copy()
        y[mask_] = (y[mask_] - self.norm_scale[class_name][0]) / (self.norm_scale[class_name][1] - self.norm_scale[class_name][0])
        y[mask_] = np.ones_like(y)[mask_] - y[mask_]
        
        if self.gt_mode == "inconsistent":
            y = torch.concat([y[:, azimuth_idx:], y[:, :azimuth_idx]], dim=-1)
            mask_ = torch.concat([mask_[:, azimuth_idx:], mask_[:, :azimuth_idx]], dim=-1)
        return rgb_obj, y, mask_
    
    
    def get_single_data(self, scene, env, epi, step, cat, scene_obj_id):
        
        class_name = category_maps[cat]
        # 加载对应data和rgb
        sampler = self.samplers[scene]
        sample_data = sampler.get_sample_multimodality(env, epi, step, ["bbsgt", "rgb", "depth", "semantic"])
        rgb = sample_data['rgb'].data
        semantic = sample_data['semantic'].data
        mask_rgb = semantic == scene_obj_id
        
        rgb_obj = rgb * mask_rgb[:,:,None]
        rgb_obj = rgb_obj.transpose((2,0,1))
        rgb_obj = self.transforms(torch.from_numpy(rgb_obj).type(torch.float32))
        
        with open(self.dir+f"/{scene}_objects_value_clip.pkl", 'rb') as f:
            object_label_clip = pickle.load(f)
        _, obj_score_arr = object_label_clip[(env, epi)][cat][scene_obj_id]

        mask_ = obj_score_arr > 1e-5        # 0 处为无值
        y = obj_score_arr.copy()
        y[mask_] = (y[mask_] - self.norm_scale[class_name][0]) / (self.norm_scale[class_name][1] - self.norm_scale[class_name][0])
        y[mask_] = np.ones_like(y)[mask_] - y[mask_]
        return rgb_obj, y, mask_
    def __len__(self):
        return len(self.data_lst)



class AzimuthDataset(Dataset):
    def __init__(
        self,
        data_lst,
        dir,
        gt_dis,
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
        
        self.gt_dis = gt_dis
        self.gt_values_data = gt_values_data
        self.gt_mode = gt_mode
        assert self.gt_mode in ["consistent", "inconsistent"], "Invalid gt mode"
        
        assert len(self.gt_dis) in [8, 11], "Invalid gt_dis"
        mode = "8dis" if len(self.gt_dis) == 8 else "11dis"
        if mode == "11dis":
            self.norm_scale = {
                'refrigerator': [10.6, 29.61],         # [12.7, 29.61],
                'couch': [11.2, 28.85],               # [13.1, 28.85],
                'toilet': [10.7, 26.83],      # [14, 26.83],
                'chair': [11.3, 30.18],       # [14.6, 30.18],
                'bed': [11.9, 28.46]
            }
        elif mode == "8dis":
            self.norm_scale = {
                'refrigerator': [12.7, 29.61],
                'couch': [13.1, 28.85],
                'toilet':  [14, 26.83],
                'chair':  [14.6, 30.18],
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
        assert y.min() > 0, "value array is < 0"
        y = torch.ones_like(y) - y
        
        # if self.gt_mode == "inconsistent":
        #     y = torch.concat([y[:, azimuth:], y[:, :azimuth]], dim=-1)
            
        # if self.augment:
        #     for yt in y_trans:
        #         y = yt(y, azimuth)
        return x, y, azimuth
    
    def __len__(self):
        return len(self.datas)
    
