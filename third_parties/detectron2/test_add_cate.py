import json

def add_categories_to_coco(input_file, output_file, new_category_names):
    # 1. 加载原始 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 获取当前最大的 category id
    if data['categories']:
        max_id = max(cat['id'] for cat in data['categories'])
    else:
        max_id = -1
    
    # 3. 增加新类别
    for name in new_category_names:
        # 检查是否已存在该类别，防止重复添加
        if any(cat['name'] == name for cat in data['categories']):
            print(f"类别 '{name}' 已存在，跳过。")
            continue
            
        max_id += 1
        new_cat = {
            "id": max_id,
            "name": name,
            # "supercategory": "none"  # COCO格式通常包含此字段，可选
        }
        data['categories'].append(new_cat)
        print(f"已添加类别: {name} (ID: {max_id})")

    # 4. 保存修改后的 JSON 数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n处理完成！新文件已保存至: {output_file}")

# --- 使用示例 ---
# 假设你的原始文件名为 'annotations.json'
new_classes = ["street light", "tree", "basketball stands", "dustbin", "statue"]  # 你想增加的新类别

add_categories_to_coco('/home/wpp/Seman-Curiosity/third_parties/detectron2/datasets/proj/annotations/instances_train_new2.json',
                       '/home/wpp/Seman-Curiosity/third_parties/detectron2/datasets/proj/annotations/instances_train_new3.json',
                       new_classes)