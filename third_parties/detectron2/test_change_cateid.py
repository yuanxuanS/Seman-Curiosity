import json

def process_coco_json(input_file, output_file):
    # 1. 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 定义 ID 映射关系
    # bench: 6 -> 2
    # building: 2 -> 3
    # grass: 3 -> 4
    # fence: 4 -> 5
    # billboard: 5 -> 6
    id_map = {
        7: 3,  # bench
        3: 4,  # building
        4: 5,  # grass
        5: 6,  # fence
        6: 7   # billboard
    }

    # 3. 处理 categories 部分
    for cat in data['categories']:
        old_id = cat['id']
        if old_id in id_map:
            cat['id'] = id_map[old_id]
            
    # 重新排序 categories（可选，按新 ID 排序更美观）
    data['categories'] = sorted(data['categories'], key=lambda x: x['id'])

    # 4. 处理 annotations 部分
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in id_map:
            ann['category_id'] = id_map[old_cat_id]

    # 5. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！已保存至: {output_file}")

# 执行处理
process_coco_json('/home/wpp/Seman-Curiosity/third_parties/detectron2/datasets/proj/annotations/instances_train_new.json', 
                  '/home/wpp/Seman-Curiosity/third_parties/detectron2/datasets/proj/annotations/instances_train_new2.json')