import os
import shutil

tgt = "/data2/wpp_data/objaverse/refrigerator"
src = "/data2/wpp_data/objaverse/glbs"
all_dirs = os.listdir(src)

cnt = 0
for dir in all_dirs:
    print(dir)
    for file in os.listdir(os.path.join(src, dir)):
        if file.endswith(".glb"):
            # print(os.path.join(src, dir, file))
            try:
                shutil.move(os.path.join(src, dir, file), os.path.join(tgt, os.path.basename(file)))
                cnt += 1
            except:
                # print(f"{file}")
                pass
            
print(cnt)