import objaverse.xl as oxl
import objaverse

import gzip
import json
from huggingface_hub import hf_hub_download

# 文件路径（假设文件名为 example.json.gz）
file_path = "/home/users/wpp/.objaverse/object-paths.json.gz"
annos = "/home/users/wpp/.objaverse/lvis-annotations.json.gz"
class_name = "refrigerator"
# 读取并解压文件
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    paths = json.load(f)  # 解析JSON数据

# 读取并解压文件
with gzip.open(annos, 'rt', encoding='utf-8') as f:
    annos_dict = json.load(f)  # 解析JSON数据
# 打印数据（可选）
# print(data.keys())        # chair, sofa, table, toilet, bed， cabinet, refrigerator, television_set, oven, microwave, sink# annotations = oxl.get_annotations(
#     download_dir="~/.objaverse" # default download directory
# )
# print(data['chair'].keys())
print(len(annos_dict[class_name]))  # ciair: 453, sofa: 81, toilet:111, bed:53,, table: 101 ,   cabinet:60, refrigerator: 55
all_paths = []
for i in range(len(annos_dict[class_name])):
    all_paths.append(paths[annos_dict[class_name][i]])
    
all_paths = sorted(all_paths)
# print(all_paths[15:20])



# 下载指定GLB模型文件

for pth in all_paths:
    file_path = hf_hub_download(
        repo_id="allenai/objaverse",
        filename=pth,  # 替换为实际文件名
        repo_type="dataset",
        local_dir_use_symlinks=False,
        local_dir="/data2/wpp_data/objaverse/"      # 必须有local_dir才能不下载软链接
    )
    print(f"模型已下载至: {file_path}")

