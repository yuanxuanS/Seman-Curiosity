import json
import pickle
import re
import torch
import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

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


rgb_info = "./third_parties/detectron2/datasets/embodied_scene/annotations/instances_train.json"
with open(rgb_info, 'r') as f:
    rgb_infos = json.load(f)

# 得到所有rgb的info
rgb_input_info = {}     # step_id: {json info}
for img_info in rgb_infos['images']:
    file_name = img_info['file_name']
    match = re.search(r'step(\d+)', file_name)
    if match:
        step_num = int(match.group(1))
        # print(step_num)  # 输出: 125
        rgb_input_info[step_num] = img_info

scene_name = "Allensville"

output_dir = f"embodied_{scene_name}"
result_info = f"/home/wpp/Seman-Curiosity/third_parties/detectron2/output/{output_dir}/inference/embodied_scene/coco_instances_results.json"
with open(result_info, "r") as f:
    result_infos = json.load(f)     
result_dict = {}
for ri in result_infos:
    result_dict[ri['image_id']] = ri

# 每个obj id 对应的rgb的step
obj_rgbs_info=f"./data_scene/visibles/info/{scene_name}.json"
with open(obj_rgbs_info, 'r') as f:
    obj_rgb_ids = json.load(f)


cate_obj = f"./data_scene/visibles/info/cate_objs_{scene_name}.pkl"
with open(cate_obj, "rb") as f:
    cate_obj_info = pickle.load(f)
# print(cate_obj_info)


cate_mapping = {
    "chair": 0,
    "couch" : 1,
    "bed": 2,
    "toilet" : 3,
    "refrigerator": 4,
}

img_dir = f"./data_scene/visibles/{scene_name}_imgs/"
second_best_loc_file = f"./{scene_name}_second_best_loc.json"
save_dir = f"./data_scene/visibles/{scene_name}_second_best/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# 要挑选的目标id
target_ids = [6,7,8]
best_locs = {}
for obj_id, rgb_ids in obj_rgb_ids.items():
    obj_id = int(obj_id)
    if not obj_id in target_ids:
        continue
    
    obj_category = None
    for name, ids in cate_obj_info.items():
        ids_ = [info[0] for info in ids]
        if obj_id in ids_:
            obj_category = name
            break
    assert obj_category is not None, "Error"
    
    cand_imgs = []
    for step in rgb_ids:
        if step not in list(rgb_input_info.keys()):
            continue
        image_id = rgb_input_info[step]['id']
        if image_id not in result_dict.keys():
            continue
        detection_result = result_dict[image_id]
        
        # whether detected category is right
        if detection_result['category_id'] ==  cate_mapping[obj_category]:
            if detection_result['score'] > 0.7:
                # best_score = detection_result['score']
                # best_img = [detection_result, step]
                # cand_imgs.append()
                rgb_name = f"epi0_env0_step{step}.png"
                rgb_img = cv2.imread(img_dir+rgb_name)
                rgb_img = visualize_coco_result(rgb_img, detection_result)
                cv2.imwrite(save_dir + f"/{step}_id{int(obj_id)}.png", rgb_img)
    
    
    # best_locs[obj_id] = [best_score, best_img]

# print(best_locs)
# with open(second_best_loc_file, "w") as f:
#     json.dump(best_locs, f)



