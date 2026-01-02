import torch
import json
from src.finetune.dataset_utils import get_loader, SampleLoader
from detectron2.structures.boxes import Boxes, BoxMode
import numpy as np
from pycocotools import mask as mask_utils
def mask_to_rle(binary_mask):
    # 1. 确保 mask 是 uint8 类型的 Numpy 数组，并且是列优先(Fortran-order)
    # 这是 pycocotools 的强制要求
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    
    # 2. 编码为 RLE 对象（包含 size 和 counts）
    rle = mask_utils.encode(fortran_mask)
    
    # 3. 此时 counts 是 bytes 类型，需要解码为字符串以便放入 JSON
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


coco_json = "./test_instances.json"     # save path
base_dir = "/home/wpp/Seman-Curiosity/data/vsqf_test_val5"
data_pth = base_dir + "/data"

from src.vqf_constants import clsid_name_maps        # TODO

CLASSES = clsid_name_maps
CLASSES_TO_IDX = {k: i for i, k in enumerate(CLASSES.keys())}
categories = [{"id": id_from_zero, "name": CLASSES[cls_id]} 
              for cls_id, id_from_zero in CLASSES_TO_IDX.items()]     # id 从0开始
coco_dict = {"info": {},
             "images": [],
             "categories": categories,
             "annotations": []}


sampler = SampleLoader(data_pth)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  

img_id = 0
anno_id = 0
for env, episode, step in zip(inputs[0], inputs[1], inputs[2]):
    sample_data = sampler.get_sample_multimodality(
        env, episode, step, ["bbsgt", "rgb", "depth"])
    
    instance = sample_data['bbsgt'].data
    if len(instance) > 0:
        rgb = sample_data['rgb'].data
        height, width = rgb.shape[:2]     #
        # print(rgb.shape)
        # depth = sample_data['depth'].data
        gt = sample_data['bbsgt']
        file_name = gt.frame.sense_info.get_path()
        y = gt.get_bbs_as_gt()
        class_labels = y.gt_classes
        class_labels = np.array(
            [CLASSES_TO_IDX[x.item()] for x in class_labels]
        )       # 从0开始的cls id
        # 
        coco_dict['images'].append(
            {
            "id": img_id,
            "file_name": f"epi{episode}_env{env}_step{step}.png",
            "width": width,
            "height": height,
            }
        )

        for id_instance in range(len(instance)): 
            # print(y[id_instance].gt_masks.numpy().shape)
            seg = mask_to_rle(y[id_instance].gt_masks.numpy()[0])
            coco_dict['annotations'].append(
                {
                    "id": int(anno_id),
                    "image_id": int(img_id),
                    "category_id": int(class_labels[id_instance]),
                    "segmentation": seg,
                    "area": int(y[id_instance].gt_masks.sum()),
                    "bbox": y[id_instance].gt_boxes.tensor[0].tolist(),
                    'bbox_mode': BoxMode.XYXY_ABS,
                    "iscrowd": 0,
                    # other key
                    "env_episode_step": [int(env), int(episode), int(step)],
                    # "object_id": env,
                    
                }
            )
            anno_id += 1
        
        img_id += 1
    # break
    
with open(coco_json, "w") as f:
    json.dump(coco_dict, f)