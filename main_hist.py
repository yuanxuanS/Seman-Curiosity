from matplotlib import pyplot as plt
import numpy as np
def calculate_cumulative_percentage(data):
    # 计算总和
    total = sum(data)
    
    # 计算每个元素的百分比
    percentages = [x / total * 100 for x in data]
    
    # 计算累积百分比
    cumulative_percentages = []
    cumulative_sum = 0
    for p in percentages:
        cumulative_sum += p
        cumulative_percentages.append(cumulative_sum)
    
    return cumulative_percentages

note = "vsqf_v3_1_eval"
l = [77, 422, 936, 1056, 953, 843, 924, 704, 439, 266, 215, 86, 5]
# [105, 538, 946, 902, 726, 390, 318, 539, 519, 396, 638, 562, 45]
# [50, 412, 1352, 1530, 1088, 890, 917, 675, 512, 231, 116, 70, 11]

cp = calculate_cumulative_percentage(l)
print(cp)
plt.plot(cp)
plt.title("cumulative percentage")
plt.savefig(f"cumulative_percentage_{note}.png")