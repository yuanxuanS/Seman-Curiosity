import json
import pickle
import re

 
rgb_info = "./third_parties/detectron2/datasets/embodied/annotations/instances_val.json"
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

     
result_info = "/home/wpp/Seman-Curiosity/third_parties/detectron2/output/embodied_Coll/inference/embodied_val/coco_instances_results.json"
with open(result_info, "r") as f:
    result_infos = json.load(f)     
result_dict = {}
for ri in result_infos:
    result_dict[ri['image_id']] = ri

obj_rgbs_info="./data/visibles/info/Collierville.json"
with open(obj_rgbs_info, 'r') as f:
    obj_rgb_ids = json.load(f)


cate_obj = "./data/visibles/info/cate_objs_Collierville.pkl"
with open(cate_obj, "rb") as f:
    cate_obj_info = pickle.load(f)
    
cate_mapping = {
    "chair": 0,
    "couch" : 1,
    "bed": 2,
    "toilet" : 3,
    "refrigerator": 4,
}

best_loc_file = "./coll_best_loc.json"
best_locs = {}
for obj_id, rgb_ids in obj_rgb_ids.items():
    obj_id = int(obj_id)
    obj_category = None
    for name, ids in cate_obj_info.items():
        ids_ = [info[0] for info in ids]
        if obj_id in ids_:
            obj_category = name
            break
    assert obj_category is not None, "Error"
    
    best_score = -10
    best_img = None
    for step in rgb_ids:
        image_id = rgb_input_info[step]['id']
        if image_id not in result_dict.keys():
            continue
        detection_result = result_dict[image_id]
        
        # whether detected category is right
        if detection_result['category_id'] ==  cate_mapping[obj_category]:
            if detection_result['score'] > best_score:
                best_score = detection_result['score']
                best_img = [detection_result, step]
    
    best_locs[obj_id] = [best_score, best_img]

print(best_locs)
with open(best_loc_file, "w") as f:
    json.dump(best_locs, f)

