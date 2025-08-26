from third_parties.Orient_Anything.utils import background_preprocess
from PIL import Image
import torchvision.transforms as T
import random
import numpy as np
pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data_vqf/bed/0a5652c16e1a4575903dfc1696382502/2.5m/render_045.png"
rgb_img = Image.open(pth).convert('RGB')
w, h = 640, 640
crop_w, crop_h = random.uniform(0.3, 1), random.uniform(0.3, 1)

seed = np.random.randint(2147483647) # make a seed with numpy generator 
random.seed(seed) # apply this seed to img transforms
transform = T.Compose([
    # T.CenterCrop((int(crop_w*w), int(crop_h*h))),
    # T.RandomResizedCrop((w,h), scale=(0.3, 1.0), ratio=(0.75, 1.33))        # scale crop大小为原来的多少倍； ratio: 长宽比
    # T.RandomHorizontalFlip()
])

from torchvision.transforms import functional as TF
transform = TF.hflip

class HorizontalFlip:

    def __init__(self, mode="consistant") -> None:
        """HorizontalFlip, 水平翻转
        """
        super().__init__()
        self.p = random.uniform(0, 1)
        self.thres = 0.5
        self.mode = mode
    def _augment_image(self, image):
        if self.p < self.thres:
            return TF.hflip(image)
        else:
            return image
    def _augment_y(self, y):
        # if self.p < self.thres:
        if self.mode == "consistent":
            return y
        elif self.mode == "inconsistent":  # y从当前角度开始，角度增加直到循环回来
            return y    # TODO
   
   
         
for i in range(10):
    transform = T.Compose([HorizontalFlip()])
    img_crop =transform._augment_image(rgb_img)
    img_crop.save(f"./t_crop{i}.png")
print(np.array(img_crop).shape)


from torchvision.transforms.functional import crop
import torch
left = int(w*random.uniform(0., 0.3))       # 不到一半
right = int(w*random.uniform(0., 0.3))
top = int(h*random.uniform(0., 0.3))
bottom = int(w*random.uniform(0., 0.3))
width = w - left - right
height = h - top - bottom
img_crop = crop(rgb_img, top, left, height, width)
# img_crop.save("./t_crop1.png")
# rm_bkg_img = background_preprocess(rgb_img, True)
# rm_bkg_img.save("./t_rmbg.png")