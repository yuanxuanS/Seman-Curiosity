from src.policy_rl.agents.utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.policy_rl.arguments import get_args
from src.finetune.dataset_utils import get_loader, SampleLoader
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer
from copy import deepcopy
from detectron2.data import MetadataCatalog
# from src.constants import coco_categories_mapping
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks

import torch
from third_parties.Orient_Anything.vision_tower import DINOv2_MLP
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug
from third_parties.Orient_Anything.utils import background_preprocess, resize_foreground
from transformers import AutoImageProcessor
from PIL import Image
from test_gt_orient import OriAny_pred, angle_diff
import numpy as np
import argparse
import yaml
import re
import os

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    72: 9,  # refrigerator
}
class_map_coco = { 
                "chair": 56,
                "couch": 57,
                "plant": 58,
                "bed": 59,
                "toilet": 61,}

class_map = { 
                "chair": 0,
                "couch": 1,
                "plant": 2,
                "bed": 3,
                "toilet": 4,}


def filter_instance(instance):
    if len(instance) == 0:
        return instance
    
    classes = instance.pred_classes
    new_instance = Instances(
                        pred_boxes=Boxes(torch.Tensor()),
                        image_size=instance.image_size,
                        pred_classes=torch.Tensor(),
                        pred_masks=torch.Tensor(),
                        scores=torch.Tensor(),
                    )
    for i in range(len(classes)):
        class_idx = classes[i]
        if class_idx in list(coco_categories_mapping.keys()):
            new_instance = new_instance.cat([instance[i]])
        else:
            continue
    return new_instance

def remove_background_detector(input_image):
    '''
    仅留下前景，4通道图像
    '''
    assert np.array(input_image).shape[2] == 4
    return input_image
def background_preprocess_detector(input_image, do_remove_background):


    if do_remove_background:
        input_image = remove_background_detector(input_image)     # rembg去掉背景，只剩前景
        input_image = resize_foreground(input_image, 0.85)      # 输入的图像有4通道，只保留前景

    return input_image

def convert_to_instance(rgb, masks, boxes_filt, pred_phrases):
    
    
    new_instance = Instances(
                        pred_boxes=Boxes(torch.Tensor()),
                        image_size=rgb.shape[:-1],
                        pred_classes=torch.Tensor(),
                        pred_masks=torch.Tensor(),
                        scores=torch.Tensor(),
                    )
    
    for mask, box, phrase in zip(masks, boxes_filt, pred_phrases):
        mask = torch.tensor(mask, dtype=torch.uint8)
        box = Boxes(box.unsqueeze(0))
        class_label = None
        for name, id in class_map_coco.items():
            if name in phrase:
                class_label = id
                break
        if class_label == None:
            break
        
        match = re.search(r'(\d+\.\d+)', phrase)
        score = float(match.group(1)) if match else 1.
        instance = Instances(
                        pred_boxes=box,
                        image_size=rgb.shape[:-1],
                        pred_classes=torch.Tensor([int(class_label)]).to(torch.int),
                        pred_masks=mask,        # BitMasks(mask),
                        scores=torch.Tensor([score]),
                    )
        
        new_instance = new_instance.cat([instance])
    return new_instance

def segment_by_segany(instance, seg_args, rgb, model, predictor, index, return_anno=False, return_per=True):
    (env, ep, step) = index
    
    new_instance = Instances(
                        pred_boxes=Boxes(torch.Tensor()),
                        image_size=rgb.shape[:-1],
                        pred_classes=torch.Tensor(),
                        pred_masks=torch.Tensor(),
                        scores=torch.Tensor(),
                    )
    
    for i in range(len(instance)):
        ins = instance[i]
        ins_name = None
        for name, id in class_map_coco.items():
            if int(ins.pred_classes[0]) == id:
                ins_name = name + "."
                break
        
        seg_args['text_prompt'] = ins_name
        masks, boxes_filt, pred_phrases = pred_segment(seg_args, rgb, model, predictor, (env, ep, step), return_anno=return_anno, return_per=return_per)
        segany_instance = convert_to_instance(rgb, masks, boxes_filt, pred_phrases)
        new_instance = new_instance.cat([segany_instance])
        
    return new_instance

if __name__ == "__main__":
    

    metadata = MetadataCatalog.get('coco_2017_val')

    
    save_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/gibson/Forkland_pred"
    base_dir = f"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env5/"
    data_pth = base_dir + "episodes_data"

    sampler = SampleLoader(data_pth)
    inputs = sampler.get_env_episode_and_steps_dense_list()     

    detector =  "maskrcnn-segany"       # "segany"     # "maskrcnn"       #
    if "maskrcnn" in detector:
        args = get_args()
        sem_pred = SemanticPredMaskRCNN(args)
    if "segany" in detector:
        gsam_config = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/configs_finetune/gsam_config/config.yaml"
        with open(gsam_config, 'r') as f:
            seg_args = yaml.load(f, Loader=yaml.FullLoader)       
        seg_args['output_dir'] = "./outputs_v2_2/"
        from src.finetune.utils.gsam_utils import pred_mask, init_segment, pred_segment
        model, predictor = init_segment(seg_args)
    
        
        
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino, val_preprocess = OriAny_pred(device)
    augment = True
    mask_and_center = False
    
    from env_orients.Cosmos_epi2 import gt_orients, No, ambig, toilet

    env = 0  # 每个环境都是 0
    eps = [2, 3, 4, 5]
    preds = []
    for ep in eps:
        for step in range(500):      # need long time
            rgb = sampler.get_sample(env, ep, step, "rgb").data
            instances_gt = sampler.get_sample(env, ep, step, "bbsgt").get_bbs_as_gt()
            
            rgb_small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # predict and segment by detector
            if detector == "maskrcnn":
                _, _, instance = sem_pred.get_prediction(rgb_small, return_score=False, return_instance=True)
                instance = filter_instance(instance)
            elif detector == "segany":
                masks, boxes_filt, pred_phrases = pred_segment(seg_args, rgb_small, model, predictor, (env, ep, step), return_anno=False, return_per=True)
                instance = convert_to_instance(rgb_small, masks, boxes_filt, pred_phrases)
            elif detector == "maskrcnn-segany":
                _, _, instance = sem_pred.get_prediction(rgb_small, return_score=False, return_instance=True)
                instance = filter_instance(instance)
                instance = segment_by_segany(instance, seg_args, rgb_small, model, predictor, (env, ep, step), return_anno=False, return_per=True)
            
            
            
            if len(instance) > 0:
                preds.append([ep, env, step])
                
                visualizer = Visualizer(
                                    deepcopy(rgb_small),
                                    metadata,
                                    instance_mode=ColorMode.IMAGE,
                                )
                frame = visualizer.draw_instance_predictions(
                                    predictions=instance.to("cpu"),
                                ).get_image()
                # cv2.imwrite(save_pth+f"/maskrcnn/ep{ep}_{step}.png", frame)
                
                pred_cls = instance.pred_classes
                pred_masks = instance.pred_masks
                pred_boxes = instance.pred_boxes
                gt_orient = gt_orients[step]
                # 
                for i, cls_id in enumerate(pred_cls):
                    
                    if augment:     # rembg库抠图， 多物体时可能漏掉？； 原图和抠图后
                        mask_ = pred_masks[i].cpu().numpy()[:, :, None]
                        rgb_obj = rgb_small * mask_
                        
                        # rgb_ = Image.fromarray(rgb_obj).convert('RGBA')
                        rgb_ = np.concatenate([rgb_obj, (mask_*255).astype(np.uint8)],axis=-1)
                        # new_alpha = np.where(mask_, 0, 1)
                        # rgb_ = np.array(rgb_)
                        # rgb_[:, :, 3] = new_alpha[:, :, 0]
                        rgb_ = Image.fromarray(rgb_)

                        rm_bkg_img = background_preprocess_detector(rgb_, True)  # 
                        angles = get_3angle_infer_aug(rgb_, rm_bkg_img, dino, val_preprocess, device)  # maskrcnn和SegmentAnything都参与投票
                        rgb_obj = np.asarray(rm_bkg_img, dtype='float')[:, :, :3]
                    else:
                        mask_ = pred_masks[i].cpu().numpy()[:, :, None]
                        rgb_obj = rgb_small * mask_
                        # rgb with mask and centerize 
                        
                        rgb_img = Image.fromarray(rgb_obj).convert('RGB')
                        angles = get_3angle(rgb_img, dino, val_preprocess, device)

                    azimuth     = float(angles[0])
                    polar       = float(angles[1])
                    rotation    = float(angles[2])
                    confidence  = float(angles[3])

                    if gt_orient == No:
                        text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, No instance in {step}"
                    elif gt_orient == ambig:
                        text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, Ambiguous instance in step {step}"
                    elif gt_orient == toilet:
                        text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, toilet instance in step {step}"
                    else:
                        gt_orient = [gt_orient] if not isinstance(gt_orient, list) else gt_orient
                        if len(gt_orient) >= len(pred_cls):
                            if gt_orient[i] != ambig and gt_orient[i] != toilet:
                                error = angle_diff(float(gt_orient[i]), azimuth)
                                text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, error: {error:.2f} in step {step}"
                            else:
                                text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, in step {step}"
                        else:
                            text = f"azimuth:{azimuth:.2f}, score:{confidence:.2f}, in step {step}"
                            
                    position = (10, 50)    # 左下角坐标
                    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
                    font_scale = 0.5       # 字体大小
                    color = (0, 255, 0)    # BGR格式颜色（绿色）
                    thickness = 2          # 线条粗细
                    rgb_obj_big = cv2.resize(rgb_obj, (640, 640), interpolation=cv2.INTER_LINEAR)
                    cv2.putText(rgb_obj_big, text, position, font, font_scale, color, thickness)
                    pth = save_pth+f"/{detector}"
                    os.makedirs(pth, exist_ok=True)
                    cv2.imwrite(pth+f"/ep{ep}_step{step}_obj{i}_cls{int(cls_id)}_oriany_aug2_b.png", rgb_obj_big)
                    
                    if mask_and_center:
                        h, w = rgb_obj.shape[:2]
                        if mask_.sum() / (h*w) < 1/9:
                            # resize and center object
                            boxes_ = pred_boxes.tensor[i].cpu().numpy()
                            x1, y1, x2, y2 = boxes_
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            object_region = rgb_obj[y1:y2, x1:x2]
                            bw, bh = x2 - x1, y2 - y1
                            
                            # 计算放大比例 (放大至图像高度的一半)
                            scale_factor = min(0.5 * h / bh, 0.5 * w / bw)
                            new_h = int(bh * scale_factor)
                            new_w = int(bw * scale_factor)
                            
                            # 
                            object_mask = mask_[y1:y2, x1:x2].astype('float')[:, :, 0]
                            object_mask_resized = cv2.resize(object_mask, (new_w, new_h)).astype('bool')
                            #  放大物体
                            
                            resized_object = cv2.resize(object_region, (new_w, new_h), 
                                                        interpolation=cv2.INTER_CUBIC)
                            #  创建居中位置
                            result = np.ones_like(rgb_obj)*255  # 纯黑背景
                            center_x = (w - new_w) // 2
                            center_y = (h - new_h) // 2
                            # 合成图像
                            result[center_y:center_y+new_h, center_x:center_x+new_w][object_mask_resized] =  resized_object[object_mask_resized]
                            
                            rgb_resized_img = Image.fromarray(result).convert('RGB')
                            angles_resized = get_3angle(rgb_resized_img, dino, val_preprocess, device)

                            azimuth_resized     = float(angles_resized[0])
                            polar_resized       = float(angles_resized[1])
                            rotation_resized    = float(angles_resized[2])
                            confidence_resized  = float(angles_resized[3])

                            if gt_orient == No:
                                text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, No instance in {step}"
                            elif gt_orient == ambig:
                                text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, Ambiguous instance in step {step}"
                            elif gt_orient == toilet:
                                text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, toilet instance in step {step}"
                            else:
                                gt_orient = [gt_orient] if not isinstance(gt_orient, list) else gt_orient
                                if len(gt_orient) >= len(pred_cls):
                                    if gt_orient[i] != ambig and gt_orient[i] != toilet:
                                        error = angle_diff(float(gt_orient[i]), azimuth_resized)
                                        text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, error: {error:.2f} in step {step}"
                                    else:
                                        text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, in step {step}"
                                else:
                                    text_rsz = f"azimuth:{azimuth_resized}, score:{confidence_resized:.2f}, in step {step}"
                                    
                            position = (10, 50)    # 左下角坐标
                            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
                            font_scale = 0.5       # 字体大小
                            color = (0, 255, 0)    # BGR格式颜色（绿色）
                            thickness = 2          # 线条粗细
                            rgb_obj_resized_big = cv2.resize(result, (640, 640), interpolation=cv2.INTER_LINEAR)
                            cv2.putText(rgb_obj_resized_big, text_rsz, position, font, font_scale, color, thickness)
                            cv2.imwrite(save_pth+f"/maskrcnn/ep{ep}_{step}_oriany_resized_white.png", rgb_obj_resized_big)
    print(preds)