import json
import matplotlib.pyplot as plt
import numpy as np

def plot_coco_category_size_distribution(json_path, save_path='size_distribution.png', y_limit=500):
    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    sorted_ids = sorted(id_to_name.keys())
    names = [id_to_name[i] for i in sorted_ids]

    # 2. 初始化统计字典 {id: {'small': 0, 'medium': 0, 'large': 0}}
    stats = {i: {'small': 0, 'medium': 0, 'large': 0} for i in sorted_ids}

    # 3. 统计尺寸
    for ann in data['annotations']:
        cat_id = ann['category_id']
        area = ann.get('area', 0)
        
        if area < 32**2:
            stats[cat_id]['small'] += 1
        elif area < 96**2:
            stats[cat_id]['medium'] += 1
        else:
            stats[cat_id]['large'] += 1

    # 准备绘图数组
    small_counts = np.array([stats[i]['small'] for i in sorted_ids])
    medium_counts = np.array([stats[i]['medium'] for i in sorted_ids])
    large_counts = np.array([stats[i]['large'] for i in sorted_ids])
    total_counts = small_counts + medium_counts + large_counts

    # 4. 绘制堆叠柱状图
    plt.figure(figsize=(14, 8))
    x = np.arange(len(names))
    
    # 绘图层叠：底部是 small，中间是 medium (加上 small 的偏移)，顶部是 large
    plt.bar(x, small_counts, label='Small (<32²)', color='#ff9999')
    plt.bar(x, medium_counts, bottom=small_counts, label='Medium', color='#66b3ff')
    plt.bar(x, large_counts, bottom=small_counts+medium_counts, label='Large (>96²)', color='#99ff99')

    # 5. 固定 Y 轴并添加标注
    plt.ylim(0, y_limit)
    
    # 在柱子顶部标注总数
    for i, total in enumerate(total_counts):
        if total > 0:
            plt.text(i, total + (y_limit * 0.01), str(total), 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 样式美化
    plt.title('Object Category Distribution by Size', fontsize=16)
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Number of Instances')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# 执行统计
plot_coco_category_size_distribution("./datasets/proj/annotations/instances_train.json", 
                                "./cate_distri_train.png")