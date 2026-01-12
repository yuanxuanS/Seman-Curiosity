from PIL import Image
import os
import numpy as np

dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data_vqf"
classes = ['bed', 'chair', 'couch', 'refrigerator', 'toilet']
# dis = ['0.5m', '1m', '1.5m', '2m','2.5m', '3m', '3.5m', '4m', '4.5m', '5m']
for cls_ in classes:
    for obj in os.listdir(dir + "/" + cls_):
        for dis in os.listdir(dir + "/" + cls_ + "/" + obj):
            for img_pth in os.listdir(dir + "/" + cls_ + "/" + obj + "/" + dis):
                # print(dir + "/" + cls_ + "/" + obj + "/" + dis + "/" + img_pth)
                pth = dir + "/" + cls_ + "/" + obj + "/" + dis + "/" + img_pth
                try:
                    img = Image.open(pth)
                except:
                    print(f"error in {pth}")
                try:
                    img = np.array(img)
                except:
                    print(f"error in {pth}")
        print(f"done {obj}")
    print(f"done {cls_}")