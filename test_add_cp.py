import os
import shutil

# 源文件
dir = "/data2/wpp_data/obj_azimuth/chair/"
# 补充文件夹
src_dir = "/data2/wpp_data/obj_azimuth/chair-bu/"
distance_added = ['0.5m', '1m', '1.5m']
# classes = ['chair']
# for cls_ in classes:
for obj in os.listdir(dir):
    # 检查是否有0.5m文件
    if '0.5m' not in os.listdir(dir + '/' + obj):
        print(f"0.5m of {obj} not exists")
        src_file = src_dir  + "/" + obj
        if not os.path.exists(src_file):
            print(f"0.5-1m {src_file} not exists in added dir")
            continue
        else:
            # 复制目标距离下的文件夹
            for dis in distance_added:
                if not os.path.exists(src_dir + '/' + obj + '/' + dis):
                    print(f"{dis} of {src_file} not exists")
                    continue
                src = src_dir + "/"+obj + "/"+dis
                dst = dir + '/' + obj + "/"+dis
                shutil.copytree(src, dst)
                print(f"copy {src} to {dst}")
        
        
        