import json
import re

Colli_best_loc = {
    31: [
        [0.14721623063087463, 0., -1.0926159620285034],
         [0.8660253882408142, 0.0, 0.5, 0.0]
         ],
    29: [
        [1.3965555429458618, 0., -5.120574474334717],
        [1.0, 0.0, 0.0, 0.0]
    ],
    38: [
        [-2.5956151485443115, 0., -0.2966674566268921],
        [-0.7071067690849304, 0.0, 0.7071067690849304, 0.0]
    ],
    10:[
        [-1.6566495895385742, 0., -4.468347072601318],
        [-0.9659258127212524, 0.0, 0.258819043636322, 0.0]
    ]
}

save_json = "./data/visibles/Collierville_tploc.json"
with open(save_json, "w") as f:
    json.dump(Colli_best_loc, f)

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

print(rgb_input_info[4708])
