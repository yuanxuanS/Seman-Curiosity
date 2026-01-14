import json
import matplotlib.pyplot as plt
from collections import Counter

def plot_coco_category_distribution(json_path, save_path='category_distribution.png'):
    # 1. 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 建立 ID 到名称的映射
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 3. 统计每个 category_id 出现的次数
    cat_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    # 4. 准备绘图数据 (按 ID 排序确保顺序一致)
    sorted_ids = sorted(id_to_name.keys())
    names = [id_to_name[i] for i in sorted_ids]
    counts = [cat_counts.get(i, 0) for i in sorted_ids]

    # 5. 绘制柱状图
    plt.figure(figsize=(12, 7))
    plt.ylim(0, 400)
    bars = plt.bar(names, counts, color='skyblue', edgecolor='navy')

    # 设置图表样式
    plt.title('Annotation Counts per Category', fontsize=15, pad=20)
    plt.xlabel('Category Name', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 6. 在柱状图上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', 
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    # 7. 保存图片
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"统计图表已保存至: {save_path}")

# 执行统计
plot_coco_category_distribution("./datasets/proj/annotations/instances_train.json", 
                                "./cate_distri_train.png")