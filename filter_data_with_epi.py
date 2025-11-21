import os
import shutil
import re

exp_dir = "/home/wpp/Seman-Curiosity/exps/dump/random_lost2/"
data_dir = exp_dir + "/episodes_data/"
episode_size = 50

target_dir = exp_dir+f"/episodes_data_{episode_size}epi/"
os.makedirs(target_dir, exist_ok=True)

cnt = 0
for file in os.listdir(data_dir):
    # match episodes
    regex = r"episode_(\d+)_step"

    match = re.search(regex, file)

    if match:
        # 1. 提取捕获组1（即括号内的数字字符串）
        episode_str = match.group(1)
        
        # 2. 将字符串转换为整数 (int)
        # Python 的 int() 函数会自动处理前导零
        episode_int = int(episode_str)
    else:
        print("error")
        
    if episode_int > episode_size:
        continue
    else:
        src_path = os.path.join(data_dir, file)
        dst_path = os.path.join(target_dir, file)
        
        shutil.copy2(src_path, dst_path)
        print(f"copy {src_path}-> {dst_path}")
        cnt += 1

print(f"move {cnt} data ")