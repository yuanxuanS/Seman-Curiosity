import pickle
from src.finetune.dataset_utils import get_loader, SampleLoader
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import os
from detectron2.data import MetadataCatalog
from copy import deepcopy
import numpy as np
from src.policy_rl.arguments import get_args
from src.policy_rl.agents.utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
import torch
import clip
from PIL import Image
from test_gt_orient import OriAny_pred
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug
from third_parties.Orient_Anything.utils import background_preprocess
from vqf_train import visualize
from src.vqf_constants import category_maps, category_id_maps, target_cls_id_in_scene

gt_angle = {"Collierville": {
    1: {1: [0, -10]},  # 0-11，15-20， 33-35
    4: {1: [180, -10]},  # 17-25
    9: {1: [260, -10]}, # 24-33
},
            "Corozal":{
    0: {1: [350, -10]},     # 15-26
    1: {1: [90, -10],
        2: [350, -10],},     # 26-35, 0-3
    3: {1: [260, -10]},     # 16-35, 0-1
    4: {1: [80, -10],
        2: [170, -10]},
    9: {1: [350, -10]}
},
            "Darden":{
    0: {1: [260, -10],
        2: [200, -10],
        4: [260, -10],
        6: [90, -10],
        },
    1: {1: [0, -10],
        2: [290, -10],
        3: [270, -10],
        },
    3: {1: [180, -10],},
    4: {1: [260, -10],},
    9: {1: [0, -10],
        3: [90, -10]}
},
            "Markleeville":{
    0: {3: [60, -10],
        4: [0, -10],},
    1: {1: [140, -10],
        },
    3: {1: [0, -10],
        },
    4: {1: [350, -10],},
    9: {1: [180, -10],
        },
},
            "Wiconisco":{
    0: {1: [200, -10],
        2: [160, -10],
        3: [170, -10], 
        4: [0, -10],        # 质量差
        6: [260, -10],
        9: [270, -10],},
    1: {1: [0, -10],},
    4: {1: [100, -10]} 
            }

}
def return_bin_idx(value, bin_center: list, bin_interval: float):
    '''
    bin_center: 
    bin_interval: bin间距的一半
    '''
    bins = [dis - bin_interval for dis in bin_center]
    bins.append(bin_center[-1] + bin_interval)
    bins = np.array(bins)
    if value < bin_center[0] - bin_interval:
        return 0
    elif value >= bin_center[-1] + bin_interval:
        return len(bin_center) - 1
    # if not isinstance(value, np.ndarray):
    #     value = np.array([value])
    return np.digitize(value, bins) - 1

scenes = ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]
metadata = MetadataCatalog.get('coco_2017_val')
pth = "/data1/wpp_data/data/obj_samples/"

mode =    "label" #"origScore"  #   "pred_orient" # "get_idx" #

# oriAny
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for scene in scenes:

    with open(pth+f"/{scene}_objects.pkl", 'rb') as f:
        data = pickle.load(f)


    data_pth = pth + "data/"+scene
    sampler = SampleLoader(data_pth)

    # save imgs
    save_dir = pth+"/obj_imgs/" + scene
    os.makedirs(save_dir, exist_ok=True)

    if mode == "origScore":
        ##  打分
        # maskrcnn
        args = get_args()
        sem_pred = SemanticPredMaskRCNN(args)
        # clip
        device = "cuda:3" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)
    elif mode == "pred_orient":
        
        object_orient = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dino, val_preprocess = OriAny_pred(device)
    elif mode == "label":
        # azimuth ground truth
        
        distances = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        angles = [i for i in range(360)]
        angle_interval = 10
        angle_bins  =angles[::10]
        object_label_maskrcnn = {}
        object_label_clip = {}
        
        cls_range_min = {i: 1e4 for i in target_cls_id_in_scene}
        cls_range_max = {i: -1 for i in target_cls_id_in_scene}
        with open(pth+f"/{scene}_objects_withorigscore.pkl", 'rb') as f:
            orig_pred = pickle.load(f)
    elif mode == "get_idx":
        distances = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        angles = [i for i in range(360)]
        angle_interval = 10
        angle_bins  =angles[::10]
        indexs = {}

            
    obj_cnt = 0
    for key, value in data.items():
        env, epi = key
        print(f"env:{env}, epi: {epi}")
        
        if mode == "label":
            object_label_maskrcnn[(env, epi)] = {}
            object_label_clip[(env, epi)] = {}
        elif mode == "pred_orient":
            object_orient[(env, epi)] = {}
        elif mode == "get_idx":
            indexs[(env, epi)] = {}
            
        for cat, cat_value in data[key].items():
            print(f"class : {cat}")

            if mode == "origScore":
                # clip
                class_name = category_maps[cat]
                text_str = [class_name]
                text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)
            elif mode == "label":
                class_name = category_maps[cat]
                class_id = category_id_maps[cat]
                object_label_maskrcnn[(env, epi)][cat] = {}
                object_label_clip[(env, epi)][cat] = {}
            elif mode == "pred_orient":
                # object_orient[(env, epi)][cat] = {}
                pass
            elif mode == "get_idx":
                indexs[(env, epi)][cat] = {}
                
            for obj, obj_value in data[key][cat].items():
                object_sample_info, scene_obj_id = data[key][cat][obj][1], data[key][cat][obj][0]
                print(f"object :{obj}, obj is in scenes: {scene_obj_id}")
                
                if scene_obj_id != None:    # None为无效物体
                    # 
                    
                    if mode == "origScore":
                        object_sample_maskrcnn = {}
                        object_sample_clip = {}
                    elif mode == "label":
                        scene_obj_id = orig_pred[key][cat][obj][0]
                        object_sample_info = orig_pred[key][cat][obj][1]
                        object_sample_maskrcnn = orig_pred[key][cat][obj][2]
                        object_sample_clip = orig_pred[key][cat][obj][3]
                        
                        object_label_maskrcnn[(env, epi)][cat][scene_obj_id] = None
                        object_label_clip[(env, epi)][cat][scene_obj_id] = None
                        score_arr_maskrcnn = np.ones((len(distances), int(360 / angle_interval)))* 1e4
                        cnt_maskrcnn = np.zeros_like(score_arr_maskrcnn)
                        score_arr_clip = np.ones((len(distances), int(360 / angle_interval)))* 1e4
                        cnt_clip = np.zeros_like(score_arr_clip)
                        
                        
                    elif mode == "pred_orient":
                        # object_orient[(env, epi)][cat][obj] = {}
                        pass
                    elif mode == "get_idx":
                        indexs[(env, epi)][cat][scene_obj_id] = {}
                        
                    for dis_range, yaw_info in object_sample_info.items():
                        dis_min, dis_max = dis_range
                        
                        if mode == "origScore":
                            object_sample_maskrcnn[dis_range] = {} 
                            object_sample_clip[dis_range] = {}
                            
                        for i, (yaw_idx, step) in enumerate(object_sample_info[dis_range].items()):

                            sample_data = sampler.get_sample_multimodality(env, epi, step, ["bbsgt", "rgb", "depth", "semantic"])
                            rgb = sample_data['rgb'].data
                            instance = sample_data['bbsgt'].data
                            depth = sample_data['depth'].data
                            semantic = sample_data['semantic'].data
                            
                            # 加载mask和对应depth，计算depth均值作为距离
                            mask_ = semantic == scene_obj_id                        
                            depth_mean= depth[mask_].mean() 
                            depth_mean = depth_mean * 4.5 + 0.5
                            
                            # 背景全黑
                            rgb_obj = rgb * mask_[:,:,None]
                            
                            # visualize
                            # visualizer = Visualizer(
                            #     deepcopy(rgb_obj),
                            #     metadata,
                            #     instance_mode=ColorMode.IMAGE,
                            # )
                            # frame = visualizer.draw_instance_predictions(
                            #         predictions=instance.to("cpu"),
                            #     ).get_image()
                            # frame = visualizer.img
                            # if i == 0:
                            # cv2.imwrite(save_dir + "/env"+str(env)+f"_epi{epi}_cls{cat}_obj{scene_obj_id}_dis{depth_mean:.2f}_yaw{yaw_idx}_first.png", frame)
                            
                            if mode == "origScore":
                                ## 打分
                                # maskrcnn
                                _, _, instance = sem_pred.get_prediction(rgb_obj, return_score=False, return_instance=True)
                                object_sample_maskrcnn[dis_range][yaw_idx] = instance

                                # clip
                                pil_img = Image.fromarray(rgb_obj)
                                image = preprocess(pil_img).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    image_features = model.encode_image(image)
                                    text_features = model.encode_text(text)
                                    
                                    logits_per_image, logits_per_text = model(image, text)
                                    score = logits_per_text.item()
                                    object_sample_clip[dis_range][yaw_idx] = score
                            elif mode == "label":
                                # 计算value
                                # maskrcnn
                                maskrcnn_ins = object_sample_maskrcnn[dis_range][yaw_idx]
                                value_maskrcnn = None
                                if len(maskrcnn_ins) == 0:
                                    value_maskrcnn = 1
                                elif len(maskrcnn_ins) == 1:
                                    if int(maskrcnn_ins.pred_classes[0].cpu().numpy()) == class_id:
                                        value_maskrcnn = 1- float(maskrcnn_ins.scores.cpu().numpy())
                                    else: # 错检
                                        value_maskrcnn = 1
                                else:
                                    value_maskrcnn = 1 - float(maskrcnn_ins.scores.mean().cpu().numpy())
                                    
                                # clip
                                clip_score = object_sample_clip[dis_range][yaw_idx]
                                value_clip = clip_score
                                
                                print(f"value: maskrcnn {value_maskrcnn}, clip {value_clip}")
                                cls_range_min[cat] = min(value_clip, cls_range_min[cat])
                                cls_range_max[cat] = max(value_clip, cls_range_max[cat])
                                
                                # 计算对应距离索引，yaw索引
                                dis_idx = return_bin_idx(depth_mean, distances, 0.25)
                                azimuth_ = gt_angle[scene][cat][obj] if (cat in list(gt_angle[scene].keys())) and (obj in list(gt_angle[scene][cat].keys())) else None
                                if azimuth_ == None:
                                    continue
                                azimuth_yaw_start, azimuth_yaw_step = azimuth_
                                
                                azimuth = yaw_idx * azimuth_yaw_step + azimuth_yaw_start
                                azimuth = azimuth % 360
                                azimuth_idx = return_bin_idx(azimuth, angle_bins, angle_interval/ 2)
                                
                                score_arr_maskrcnn[dis_idx, azimuth_idx] = value_maskrcnn
                                cnt_maskrcnn[dis_idx, azimuth_idx] += 1
                                score_arr_clip[dis_idx, azimuth_idx] = value_clip if cnt_clip[dis_idx, azimuth_idx] < 1e-5 else score_arr_clip[dis_idx, azimuth_idx] + value_clip
                                cnt_clip[dis_idx, azimuth_idx] += 1
                            elif mode == "get_idx":
                                # 计算对应距离索引，yaw索引
                                dis_idx = return_bin_idx(depth_mean, distances, 0.25)
                                azimuth_ = gt_angle[scene][cat][obj] if (cat in list(gt_angle[scene].keys())) and (obj in list(gt_angle[scene][cat].keys())) else None
                                if azimuth_ == None:
                                    continue
                                azimuth_yaw_start, azimuth_yaw_step = azimuth_
                                
                                azimuth = yaw_idx * azimuth_yaw_step + azimuth_yaw_start
                                azimuth = azimuth % 360
                                azimuth_idx = return_bin_idx(azimuth, angle_bins, angle_interval/ 2)

                                indexs[(env, epi)][cat][scene_obj_id][step] = [dis_idx, azimuth_idx]
                            elif mode == "pred_orient":
                                rgb_img = Image.fromarray(rgb_obj).convert('RGB')
                                rm_bkg_img = background_preprocess(rgb_img, True)
                                angles = get_3angle_infer_aug(rgb_img, rm_bkg_img, dino, val_preprocess, device)
                                
                                azimuth     = float(angles[0])
                                polar       = float(angles[1])
                                rotation    = float(angles[2])
                                confidence  = float(angles[3])
                                
                                object_orient[(env, epi)][step] = azimuth
                                print(f"env {env} epi {epi}, yawidx: {yaw_idx}, step: {step}, azimuth: {azimuth}")
                                cv2.imwrite(save_dir + "/env"+str(env)+f"_epi{epi}_step{step}_yaw{yaw_idx}_azimuth{azimuth:.2f}_.png", rgb_obj)
                    
                        
                                
                    obj_cnt += 1
                    if mode == "origScore":
                        data[key][cat][obj] = [scene_obj_id, 
                                            object_sample_info, 
                                            object_sample_maskrcnn,
                                            object_sample_clip]
                    elif mode == "label":
                        print(f"has {(cnt_maskrcnn > 0).sum()} prediction with maskrcnn ")
                        print(f"has {(cnt_clip > 0).sum()} prediction with clip ")
                        score_arr_maskrcnn[score_arr_maskrcnn> 1e3] = 0.
                        
                        # 先求平均值，再去掉无值的元素
                        score_arr_maskrcnn[cnt_maskrcnn > 0] = score_arr_maskrcnn[cnt_maskrcnn > 0] / cnt_maskrcnn[cnt_maskrcnn > 0]
                        score_arr_clip[cnt_clip > 0] = score_arr_clip[cnt_clip > 0] / cnt_clip[cnt_clip > 0]
                        
                        score_arr_clip[score_arr_clip> 1e3] = 0.
                        
                        
                        
                        
                        object_label_maskrcnn[(env, epi)][cat][scene_obj_id] = [step, score_arr_maskrcnn]
                        object_label_clip[(env, epi)][cat][scene_obj_id] = [step, score_arr_clip.copy()]
                        
                        if (score_arr_clip > 0).sum() > 0:
                            score_arr_clip[score_arr_clip > 0] = (score_arr_clip[score_arr_clip > 0] - score_arr_clip[score_arr_clip > 0].min()) / (score_arr_clip[score_arr_clip > 0].max() - score_arr_clip[score_arr_clip > 0].min())
                            score_arr_clip[score_arr_clip > 0] = 1 - score_arr_clip[score_arr_clip > 0]
                            # 绘图
                            text = f"env{env}_epi{epi}_step{step}"
                            visualize(score_arr_maskrcnn, scene, pth+"/vis", text+"_maskrcnn.png")
                            visualize(score_arr_clip, scene, pth+"/vis", text+"_clip.png")
                        print(f"maskrcnn: ", score_arr_maskrcnn[score_arr_maskrcnn > 0])
                        print(f"clip: ", score_arr_clip[score_arr_clip > 0])
                        
                        
                    elif mode == "pred_orient":
                        # object_orient[(env, epi)][cat][obj] = angles
                        pass
                    # elif mode == "pred_orient"
                        
    if mode == "origScore":
        with open(pth+f"/{scene}_objects_withorigscore.pkl", 'wb') as f:
            pickle.dump(data, f)
    elif mode == "pred_orient":
        with open(pth+f"/{scene}_objects_withorient.pkl", 'wb') as f: 
            pickle.dump(object_orient, f)
    elif mode == "label":
        with open(pth+f"/{scene}_objects_value_maskrcnn.pkl", 'wb') as f:
            pickle.dump(object_label_maskrcnn, f)
        with open(pth+f"/{scene}_objects_value_clip.pkl", 'wb') as f:
            pickle.dump(object_label_clip, f)
        print(f"clip min: {cls_range_min}, max: {cls_range_max}")
    elif mode == "get_idx":
        with open(pth+f"/{scene}_objects_index.pkl", 'wb') as f:
            pickle.dump(indexs, f)
    print(f"finish {scene}")