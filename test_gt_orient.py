import torch
import torch.nn.functional as F
from third_parties.Orient_Anything.vision_tower import DINOv2_MLP
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug
from third_parties.Orient_Anything.utils import background_preprocess
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer
from copy import deepcopy
from detectron2.data import MetadataCatalog
import copy
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks

metadata = MetadataCatalog.get('coco_2017_val')

cls_id_map = {0: 57, 1:58, 2:59, 3:61, 4:62, 5:60, 6:6}

def map_to_original_cls_gt(instance, cls):
    '''
    cls: {0: 57, 1:58, ...}
    '''
    new_instance = copy.deepcopy(instance)

    for i in range(len(instance.gt_classes)):
        new_idx = cls[int(instance.gt_classes[i])]
        new_instance.gt_classes[i] = new_idx
    new_instance.gt_classes = torch.tensor(new_instance.gt_classes)
    return new_instance
  
def resize_instance(instance, tgt_size):
    '''
    resize instance to tgt_size
    '''
    new_instance = Instances(
                    image_size=tgt_size,
                    gt_boxes=instance.gt_boxes,
                    gt_classes=instance.gt_classes,
                    gt_masks=instance.gt_masks,
                )
    new_instance.gt_boxes.tensor[:, 0::2] = new_instance.gt_boxes.tensor[:, 0::2] * tgt_size[0] / instance.image_size[0]
    new_instance.gt_boxes.tensor[:, 1::2] = new_instance.gt_boxes.tensor[:, 1::2] * tgt_size[1] / instance.image_size[1]
    new_instance.gt_boxes.tensor[:, 0::2] = torch.clamp(new_instance.gt_boxes.tensor[:, 0::2], min=0)
    new_instance.gt_boxes.tensor[:, 1::2] = torch.clamp(new_instance.gt_boxes.tensor[:, 1::2], min=0)
    new_instance.gt_boxes.tensor[:, 0::2] = torch.clamp(new_instance.gt_boxes.tensor[:, 0::2], max=tgt_size[0])
    new_instance.gt_boxes.tensor[:, 1::2] = torch.clamp(new_instance.gt_boxes.tensor[:, 1::2], max=tgt_size[1])
    new_instance.gt_masks = F.interpolate(instance.gt_masks[None, ...].float(), size=tgt_size, mode='bilinear', align_corners=False).bool()[0]
    return new_instance
  

def OriAny_pred(device):
  # Orient anything
  ckpt_path = "./third_parties/Orient_Anything/models/largeEx2/dino_weight.pt"


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

  val_preprocess   = AutoImageProcessor.from_pretrained("/home/wpp/Seman-Curiosity/third_parties/Orient_Anything/models/dino/", local_files_only=True)
  return dino, val_preprocess

def angle_diff(theta1, theta2):
    diff = theta2 - theta1
    normalized_diff = (diff + 180) % 360 - 180
    return abs(normalized_diff)

# # 示例
# θ1, θ2 = 1, 359
# diff = angle_diff(θ1, θ2)  # 输出 -2
# min_diff = abs(diff)       # 最小差值为 2°


def visualize(grid_count, all_images, all_azimuths, all_scores, all_name, all_image_pred, note, epi):
  rows = 2
  cols = 5
  images_per_grid = rows * cols
  margin = 60  # 底部留白
  border_width = 3  # 边框宽度
  border_color = (200, 200, 200)  # 浅灰色边框
  font_scale = 0.8  # 字体大小
  font_thickness = 2  # 字体粗细
  font_color = (0, 0, 0)  # 黑色字体


  # 按照错误率分别可视化
  for i in range(0, len(all_images), 10):
    current_images = all_images[i:i+10]
    current_images_pred = all_image_pred[i:i+10]
    current_azimuths = all_azimuths[i:i+10]
    current_scores = all_scores[i:i+10]
    current_names = all_name[i:i+10]
    # 计算实际需要处理的图像数量
    num_images = len(current_images)
    # 计算需要的行数（可能不足10张）
    actual_rows = 2 if num_images > cols else 1
    actual_cols = min(cols, num_images)
    # 获取单张图像尺寸
    img_height, img_width = current_images[0].shape[:2]
    img_height *= 2
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
      grid_img[y_offset+border_width:y_offset+int(img_height/2)-border_width, 
              x_offset+border_width:x_offset+img_width-border_width] = current_images[idx][border_width:-border_width, border_width:-border_width]
      
      grid_img[y_offset+int(img_height/2)+border_width:y_offset+img_height-border_width,
              x_offset+border_width:x_offset+img_width-border_width] = current_images_pred[idx][border_width:-border_width, border_width:-border_width]
              
      # 添加分数标注
      score_text = f"Allen_epi{ep}_step{current_names[idx][2]}" 
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
    cv2.imwrite(f"./images/gibson/Allensville_epi{ep}/OriAny_{grid_count}_{note}.png", grid_img)
    # cv2.imwrite(f"./images/gibson/Allensville_epi1/step{current_names[idx][2]}.png", grid_img)
    grid_count += 1
        
    # 重置当前批次
    current_images = []
    current_images_pred = []
    current_scores = []
    current_azimuths = []
    current_names = []

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dino, val_preprocess = OriAny_pred(device)

  # 遍历每张图像, 对比gt和OriAny检测结果，超出30度、60度分别打印；
  base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env1/"
  data_pth = base_dir + "episodes_data"
  from src.finetune.dataset_utils import get_loader, SampleLoader
  sampler = SampleLoader(data_pth)

  inputs = sampler.get_env_episode_and_steps_dense_list()     


  # errors
  err_30 = []
  err_30_60 = []
  err_60 = []


  # 存储当前批次的图像和分数
  all_images_err30 = []
  all_scores_err30 = []
  all_azimuths_err30 = []
  all_name_err30 = []
  all_images_pred_err30 = []
  all_images_pred_err30_note = []

  all_images_err30_60 = []
  all_scores_err30_60 = []
  all_azimuths_err30_60 = []
  all_name_err30_60 = []
  all_images_pred_err30_60 = []
  all_images_pred_err30_60_note = []

  all_images_err60 = []
  all_scores_err60 = []
  all_azimuths_err60 = []
  all_name_err60 = []
  all_images_pred_err60 = []
  all_images_pred_err60_note = []
  pred_img = []

  env = 0
  ep = 3

  mode = ""   # "draw_pred"   # "gt"都没有感觉没有feel
  augment = False
  
  from Allensville_epi3_ import gt_orients, No, ambig, toilet

  step_list = []

  for step in inputs[2]:      # need long time
      instances = sampler.get_sample(env, ep, step, "bbsgt").get_bbs_as_gt()
      instances_pred = sampler.get_sample(env, ep, step, "bbs").get_bbs_as_gt()

      if len(instances) > 0:
        if len(instances_pred) > 0:
          pred_img.append(step)
        gt_orient = gt_orients[step]
        assert gt_orient != No, f"step: {step} has instance but gt_orient is No"
            
        gt_orient = [gt_orient] if not isinstance(gt_orient, list) else gt_orient
        
        gt_cls = instances.gt_classes
        gt_masks = instances.gt_masks
        assert len(gt_cls) == len(gt_orient), f"gt orient and clas not consistent in step {step}"
        for i, cls_id in enumerate(gt_cls):
          if gt_orient[i] != No:
            if gt_orient[i] != ambig:
              if gt_orient[i] != toilet:
                assert isinstance(gt_orient[i], int), f"gt_orient is {gt_orient[i]}"
                # pred by OriAny
                # rgb with background
                rgb = sampler.get_sample(env, ep, step, "rgb").data
                
                if mode == "draw_pred":
                  # with pred instance
                  rgb_small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

                  visualizer = Visualizer(
                            deepcopy(rgb_small),
                            metadata,
                            instance_mode=ColorMode.IMAGE,
                        )
                  # instances_pred = map_to_original_cls_gt(instances_pred.to('cpu'), cls_id_map)
                  # if len(instances_pred) > 0:
                  #   instances_pred = resize_instance(instances_pred, rgb.shape[:2])
                  frame = visualizer.draw_instance_gt(
                            predictions=instances_pred.to("cpu"),
                        ).get_image()
                  frame_big = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                else:
                  frame_big = rgb
                # rgb with mask
                mask_ = gt_masks[i].cpu().numpy()[:, :, None]
                rgb_obj = rgb * mask_
                # rgb with mask and centerize 
                
                rgb_img = Image.fromarray(rgb_obj).convert('RGB')
                if augment:
                  rm_bkg_img = background_preprocess(rgb_img, True)
                  angles = get_3angle_infer_aug(rgb_img, rm_bkg_img, dino, val_preprocess, device)
                else:
                  angles = get_3angle(rgb_img, dino, val_preprocess, device)

                azimuth     = float(angles[0])
                polar       = float(angles[1])
                rotation    = float(angles[2])
                confidence  = float(angles[3])

                error = angle_diff(float(gt_orient[i]), azimuth)
                # print(f' azimuth: {azimuth}, error: {error}, confidence: {confidence}')

                if error <= 30:
                  err_30.append(step)
                  all_images_err30.append(rgb_obj)
                  all_azimuths_err30.append(azimuth)
                  all_scores_err30.append(confidence)
                  all_name_err30.append([ep, env, step])
                  all_images_pred_err30.append(frame_big)
                else:
                  if error <= 60:
                    err_30_60.append(step)
                    all_images_err30_60.append(rgb_obj)
                    all_azimuths_err30_60.append(azimuth)
                    all_scores_err30_60.append(confidence)
                    all_name_err30_60.append([ep, env, step])
                    all_images_pred_err30_60.append(frame_big)
                  else:
                    err_60.append(step)
                    all_images_err60.append(rgb_obj)
                    all_azimuths_err60.append(azimuth)
                    all_scores_err60.append(confidence)
                    all_name_err60.append([ep, env, step])
                    all_images_pred_err60.append(frame_big)
                
                
                # masked object
                # centerize masked object
                # centerize masked object and resize to 224 (object 3/4)

  # 统计比例
  print(f"error <30: {len(err_30)} \n error 30-60: {len(err_30_60)} \n error >60: {len(err_60)}")
  print(pred_img)

  grid_count = 0
  name = "err30_aug" if augment else "err30"
  visualize(grid_count, all_images_err30, 
            all_azimuths_err30, 
            all_scores_err30, 
            all_name_err30,
            all_images_pred_err30,
            name, ep)

  grid_count = 0
  name = "err30_60_aug" if augment else "err30_60"
  visualize(grid_count, all_images_err30_60, 
            all_azimuths_err30_60, 
            all_scores_err30_60, 
            all_name_err30_60,
            all_images_pred_err30_60,
            name, ep)

  grid_count = 0
  name = "err60_aug" if augment else "err60"
  visualize(grid_count, all_images_err60, 
            all_azimuths_err60, 
            all_scores_err60, 
            all_name_err60,
            all_images_pred_err60,
            name, ep)