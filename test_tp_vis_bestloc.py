import json
import os
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def visualize_coco_result(rgb_img, coco_result, metadata_name="my_dataset"):
    """
    rgb_img: numpy array (H, W, 3)
    coco_result: 你提供的 list 结果
    """
    height, width = rgb_img.shape[:2]
    
    # 1. 初始化 Instances 对象
    instances = Instances((height, width))
    
    bboxes = []
    classes = []
    scores = []
    masks = []

    # for res in coco_result:
        # Bbox 转换: [x, y, w, h] -> [x1, y1, x2, y2]
    res = coco_result
    x, y, w, h = res['bbox']
    bboxes.append([x, y, x + w, y + h])
    
    # 类别和分数
    classes.append(res['category_id'])
    scores.append(res['score'])
    
    # 掩码解码 (RLE -> Binary Mask)
    mask = mask_util.decode(res['segmentation'])
    masks.append(mask)

    # 2. 赋值到 Instances 属性
    instances.pred_boxes = Boxes(torch.tensor(bboxes))
    instances.pred_classes = torch.tensor(classes)
    instances.scores = torch.tensor(scores)
    instances.pred_masks = torch.tensor(np.stack(masks)).bool()

    # 3. 可视化绘制
    # metadata = MetadataCatalog.get(metadata_name)
    # 假设你的类别 4 是 'target_object'
    # metadata.thing_classes = ["class0", "class1", "class2", "class3", "target_object"] 
    
    visualizer = Visualizer(rgb_img,)
    vis_output = visualizer.draw_instance_predictions(instances)
    
    return vis_output.get_image()


best_loc_file = "./coll_best_loc.json"
img_dir = "./data/visibles/Collierville_imgs/"
save_dir = "./data/visibles/Collierville_best/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(best_loc_file, "r") as f:
    best_locs = json.load(f)
# print(best_locs['3'][1][0])
for obj_id, info in best_locs.items():
    if info[0] < 0:     # 不存在看到的
        continue
    
    
    step = info[1][-1]
    rgb_name = f"epi0_env0_step{step}.png"
    
    rgb_img = cv2.imread(img_dir+rgb_name)
    # print(obj_id,info)
    rgb_img = visualize_coco_result(rgb_img, info[1][0])
    cv2.imwrite(save_dir + f"/{step}.png", rgb_img)