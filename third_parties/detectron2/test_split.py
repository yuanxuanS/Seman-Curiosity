import json
import random
import os

def split_coco_dataset(json_path, train_json_path, test_json_path, train_ratio=0.8):
    """
    json_path: 原始合并后的完整 json 路径
    train_json_path: 训练集保存路径
    test_json_path: 测试集保存路径
    train_ratio: 训练集所占比例 (0.0 - 1.0)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    info = data.get('info', {})

    # 1. 随机打乱图片列表
    random.shuffle(images)

    # 2. 计算分割点
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # 3. 根据图片 ID 提取对应的标注
    # 创建图片 ID 的集合以提高查找效率
    train_img_ids = set(img['id'] for img in train_images)
    test_img_ids = set(img['id'] for img in test_images)

    train_annotations = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_img_ids]

    # 4. 构建新的字典结构
    train_data = {
        "info": info,
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }

    test_data = {
        "info": info,
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories
    }

    # 5. 保存文件
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"分割完成！")
    print(f"训练集: {len(train_images)} 张图片, {len(train_annotations)} 个标注")
    print(f"测试集: {len(test_images)} 张图片, {len(test_annotations)} 个标注")

# 使用示例
split_coco_dataset(
    json_path='./datasets/proj/annotations/final_merged_dataset.json', 
    train_json_path='./datasets/proj/annotations/instances_train.json', 
    test_json_path='./datasets/proj/annotations/instances_test.json', 
    train_ratio=0.5
)