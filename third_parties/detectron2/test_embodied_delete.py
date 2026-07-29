import json
import random
import os

def delete_random_images_from_coco(input_json, output_json, n_to_delete):
    """
    input_json: 原始标注文件路径
    output_json: 删除后的新文件路径
    n_to_delete: 准备删除的图像数量
    """
    
    # 1. 加载 JSON 数据
    with open(input_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    
    # 安全检查：如果要删除的数量超过总数，则清空
    if n_to_delete >= len(images):
        print(f"警告：要删除的数量 ({n_to_delete}) 大于或等于总图数 ({len(images)})")
        coco_data['images'] = []
        coco_data['annotations'] = []
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=4)
        return

    # 2. 随机选择要【删除】的图像对象
    # random.sample 直接从列表中不重复地抽取
    images_to_remove = random.sample(images, n_to_delete)
    
    # 获取这些被删图像的 ID 集合，方便后续过滤标注
    removed_ids = {img['id'] for img in images_to_remove}
    removed_names = [img['file_name'] for img in images_to_remove]

    # 3. 筛选【保留】的图像和标注
    # 保留 ID 不在 removed_ids 中的图像
    remaining_images = [img for img in images if img['id'] not in removed_ids]
    
    # 保留 image_id 不在 removed_ids 中的标注
    remaining_annotations = [ann for ann in annotations if ann['image_id'] not in removed_ids]

    # 4. 更新 COCO 结构
    new_coco = {
        "info": coco_data.get("info", {}),
        "images": remaining_images,
        "categories": coco_data.get("categories", []),
        "annotations": remaining_annotations
    }

    # 5. 保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, indent=4, ensure_ascii=False)

    print(f"处理完成！")
    print(f"原始图像数: {len(images)} | 删除图像数: {n_to_delete} | 剩余图像数: {len(remaining_images)}")
    print(f"剩余标注条数: {len(remaining_annotations)}")
    print(f"随机删除的前 3 张图示例: {removed_names[:3]}")

# --- 配置 ---
original_json = "./datasets/embodied_sequencev2_wotrajR_bg4_r3_rnd/annotations/instances_train_epi15.json" # 输入文件
new_subset_json = "./datasets/embodied_sequencev2_wotrajR_bg4_r3_rnd/annotations/instances_train_epi15_1.json" # 输出文件
N = 4317 # 想要随机删除 10 张图的标注

delete_random_images_from_coco(original_json, new_subset_json, N)