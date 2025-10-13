import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def return_bin_idx(value, bin_center: list, bin_interval: float):
    '''
    bin_center: 
    bin_interval: bin间距的一半
    '''
    bins = [dis - bin_interval for dis in bin_center]
    bins.append(bin_center[-1] + bin_interval)
    bins = np.array(bins)
    if value < bin_center[0] - bin_interval:
        return 0
    elif value >= bin_center[-1] + bin_interval:
        return len(bin_center) - 1
    # if not isinstance(value, np.ndarray):
    #     value = np.array([value])
    return np.digitize(value, bins) - 1

def visualize(x, obj_str, save_dir, note=""):
    '''
    x : h*w
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    # 绘制热力图
    plt.figure(figsize=(18, 3))  # 宽度调大以适应360列
    im = plt.imshow(x, 
                    cmap='hot',  # 颜色映射
                    aspect='auto',   # 自动调整纵横比防止变形
                    interpolation='none')  # 禁用插值保持清晰边界
    plt.colorbar(im, label='Value')  # 添加颜色条
    plt.title('value Heatmap')
    plt.xlabel('azimuth (0-359)')
    plt.ylabel('distance')
    os.makedirs(save_dir+"/"+obj_str, exist_ok=True)
    plt.savefig(save_dir+"/"+obj_str+"/"+note+".png", dpi=300, bbox_inches='tight')  # 保存高清图
    plt.show()