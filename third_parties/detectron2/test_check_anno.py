# 伪代码：检查 json 中是否存在空的 segmentation
import json
data = json.load(open("./datasets/proj/annotations/instances_test.json"))
for ann in data['annotations']:
    if 'segmentation' in ann:
        if not ann['segmentation'] or len(ann['segmentation']) == 0:
            print(f"发现异常标注，ID: {ann['id']} \n {ann['image_id']}")