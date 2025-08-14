from src.policy_rl.agents.utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.policy_rl.arguments import get_args
from src.finetune.dataset_utils import get_loader, SampleLoader
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer
from copy import deepcopy
from detectron2.data import MetadataCatalog
from src.constants import coco_categories_mapping
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
import torch
from third_parties.Orient_Anything.vision_tower import DINOv2_MLP
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug
from third_parties.Orient_Anything.utils import background_preprocess, resize_foreground
from transformers import AutoImageProcessor
from PIL import Image
from test_gt_orient import OriAny_pred, angle_diff
import numpy as np
import argparse


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


if __name__ == "__main__":
    detector = "segany"     # "maskrcnn"
    if detector == "segany":
        parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
        parser.add_argument("--config", type=str, required=True, help="path to config file")
        parser.add_argument(
            "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
        )
        parser.add_argument(
            "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
        )
        parser.add_argument(
            "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
        )
        parser.add_argument(
            "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
        )
        parser.add_argument(
            "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
        )
        parser.add_argument("--input_image", type=str, required=True, help="path to image file")
        parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
        parser.add_argument(
            "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
        )

        parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
        parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

        parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
        parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
        args = parser.parse_args()
    metadata = MetadataCatalog.get('coco_2017_val')

    
    save_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/gibson/Beechwood_pred"
    base_dir = f"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env2/"
    data_pth = base_dir + "episodes_data"

    sampler = SampleLoader(data_pth)
    inputs = sampler.get_env_episode_and_steps_dense_list()     

    
    if detector == "maskrcnn":
        args = get_args()
        sem_pred = SemanticPredMaskRCNN(args)
    elif detector == "segany":
         

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino, val_preprocess = OriAny_pred(device)
    augment = True
    mask_and_center = False
    
    from Cosmos_epi2 import gt_orients, No, ambig, toilet

    env = 0  # 每个环境都是 0
    ep = 4
    preds = []
    for step in range(500):      # need long time
        rgb = sampler.get_sample(env, ep, step, "rgb").data
        instances_gt = sampler.get_sample(env, ep, step, "bbsgt").get_bbs_as_gt()
        
        rgb_small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

        # predict by maskrcnn
        semantic_pred, rgb_vis, instance = sem_pred.get_prediction(rgb_small, return_score=False, return_instance=True)
        instance = filter_instance(instance)
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
                cv2.imwrite(save_pth+f"/maskrcnn/ep{ep}_{step}_obj{i}_oriany_aug2.png", rgb_obj_big)
                
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