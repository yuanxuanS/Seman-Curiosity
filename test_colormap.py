import matplotlib.pyplot as plt
import numpy as np

# 查看所有可用的colormap名称
# print(plt.colormaps()) # 会输出一个很长的列表，包括 'viridis', 'plasma', 'inferno' 等[5,7](@ref)

# 获取特定的colormap对象，例如'viridis'
cmap = plt.get_cmap('Blues')  # 或者使用 cmap = plt.cm.viridis

# 查看colormap上特定位置（例如中点0.5）的RGBA值（范围0-1）
rgba_value = cmap(0.5) # 参数为归一化到[0,1]之间的数值
print(f"中点0.5的RGBA值 (0-1范围): {rgba_value}")

# 如果你有一个0-255范围的整数索引（如200），也可以直接传入
# colormap会自动将其归一化到[0,1]：x/255.0
rgba_value_255 = cmap(200)
print(f"索引200的RGBA值: {rgba_value_255}")

# 如果你想获取整个colormap的256个颜色（默认长度）
num_colors = 15
colors = cmap(np.linspace(0, 1, num_colors)) # 得到一个(256, 4)的数组
print(f"整个colormap的RGB数组形状: {colors.shape}")
# 打印前5个颜色看看
print("前5个颜色 (RGBA):")
print(colors)