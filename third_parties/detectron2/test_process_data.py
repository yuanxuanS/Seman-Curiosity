import json

offset = 853 # 432      # 134
file = "./instance3.json"     # "./instance.json"
file_new = "./instance_new_3.json"
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
print(ids)