import os
import shutil
import re
import pickle

dir = "/data2/wpp_data/obj_azimuth/toilet/"
objs = [
"3adba192e0dc45c29e37356776cbc228"
]

# fix x
# tmp_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/tmp/"
# distances = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
# for obj in objs:
#     for distance in distances:
#         for img in os.listdir(dir + obj + "/" + distance):
#             # 修改图像名字
#             match = re.search(f'\d+', img)
#             if match:
#                 number = int(match.group())
#                 re_number = (number + 180) % 360
#                 src_file = dir + obj + "/" + distance + "/" + img
#                 dst_file = tmp_dir + obj + "/" + distance + "/" + f"render_{re_number:03d}.png"
#                 os.makedirs(tmp_dir + obj + "/" + distance + "/", exist_ok=True)
#                 shutil.copy(src_file, dst_file)
#                 print(f"cp and rename {src_file} to {dst_file}")

# fix y
dir_y = "/data2/wpp_data/obj_azimuth/"
with open(dir_y + "toilet_clip_add_orig.pkl", "rb") as f:
    data = pickle.load(f)
    # print(data)
    
from test_label import draw_array
distances = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
for obj in objs:
    draw_array(obj, data[obj], obj+"_before", True, "./images/tmp")
    for dis in distances:
        data[obj][dis] = data[obj][dis][-180:] + data[obj][dis][:-180]
    draw_array(obj, data[obj], obj+"_re", True, "./images/tmp")

with open(dir_y + "toilet_clip_fix.pkl", "wb") as f:
    pickle.dump(data, f)