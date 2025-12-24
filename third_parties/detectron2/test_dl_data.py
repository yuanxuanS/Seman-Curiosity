import json

# f= "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/third_parties/detectron2/datasets/coco/annotations/instances_val2017.json"
# with open(f, "r") as f:
#     data = json.load(f)

# # print(data['images'][0])
# print(data.keys())
# print(data['annotations'][2]['segmentation'])


import json
import os

import json

file = "./datasets/proj/annotations/instances_train.json"

with open(file, "r") as f:
    data = json.load(f)

# 将所有标注的 iscrowd 改为 0
for ann in data["annotations"]:
    ann["iscrowd"] = 0

with open("fixed_annotations.json", "w") as f:
    json.dump(data, f)
