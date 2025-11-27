import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib

patch_h = 16
patch_w = 16
feat_dim = 768

transform = T.Compose([
    # T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dinov2_vitb14 = torch.hub.load('', 'dinov2_vitb14',source='local').cuda()

features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()

img_path = f'/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/frontier_env1/imgs/epi1_env0_step0.png'
img = Image.open(img_path).convert('RGB')
imgs_tensor[0] = transform(img)[:3]

with torch.no_grad():
    features_dict = dinov2_vitb14.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()
print(f"feature shape {features.shape}")