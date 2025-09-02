import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
def get_grid(pose, grid_size, device):
    """
    fanhu
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    # 二维旋转矩阵： 向量和x轴(向右)的夹角为t
    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    # 平移矩阵
    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))     # 目标图像的像素在原图像中的采样位
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid

    

# 生成测试地图（带对角线的正方形）
map_tensor = torch.zeros(1, 1, 200, 200)
for i in range(200):
    map_tensor[0, 0, i, i] = 1.0       # 主对角线
    map_tensor[0, 0, i, 199-i] = 0.5   # 副对角线

# 应用30度旋转 + 右移50像素（x,y范围为[-1, 1], x方向水平， x>0, 向左移动；y>0, 向上； 逆时针旋转角度 ）
pose = torch.tensor([0, 0.5, 30]).unsqueeze(0)     
rot_grid, trans_grid = get_grid(pose, map_tensor.size(), "cpu")     # 

# grid = F.affine_grid(theta, map_tensor.size(), align_corners=False)
rotated = F.grid_sample(map_tensor, rot_grid, align_corners=True)
translated = F.grid_sample(rotated, trans_grid, align_corners=True)
# translated_only = F.grid_sample(map_tensor, trans_grid, align_corners=True)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(map_tensor.squeeze(), cmap='viridis')
ax[0].set_title("Original")
ax[1].imshow(translated.squeeze().detach(), cmap='viridis')
ax[1].set_title("Rotated 30° + Right Shift")
# plt.show()
plt.savefig('test_valuemap_affine.png')