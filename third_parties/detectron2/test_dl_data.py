import json

# f= "/home/wpp/Seman-Curiosity/third_parties/detectron2/datasets/coco/annotations/instances_val2017.json"
# with open(f, "r") as f:
#     data = json.load(f)

# print(data['images'][0])
# print(data.keys())
# print(data['annotations'][0])


# import json
# import os

import json

file = "./anno/frontier_wpp_new.json"

with open(file, "r") as f:
    data = json.load(f)

# 将所有标注的 iscrowd 改为 0
for ann in data["annotations"]:
    ann["iscrowd"] = 0

with open("./anno/frontier_wpp_new2.json", "w") as f:
    json.dump(data, f)
