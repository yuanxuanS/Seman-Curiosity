import json

offset = 545 # 432      # 134
file = "./anno/lyr-3.json"     # "./instance.json"
file_new = "./anno/lyr-3-new.json"
with open(file, "r") as f:
    data = json.load(f)

## id 偏置
for img in data['images']:
    img['id'] += offset

for ann in data['annotations']:
    ann['image_id'] += offset
    

with open(file_new, "w") as f:
    json.dump(data, f)

with open(file_new, "r") as f:
    data = json.load(f)

ids = []
for img in data['images']:
    ids.append(img['id'])
print(max(ids), min(ids))