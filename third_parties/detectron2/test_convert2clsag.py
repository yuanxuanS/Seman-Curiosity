import json

def convert_coco_to_single_class(input_json_path, output_json_path, new_class_name="object"):
    # 1. 加载原始 JSON 数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 定义新的单一类别 ID (通常设为 1 或 0)
    target_category_id = 0
    
    # 3. 重新构造 categories
    data['categories'] = [
        {
            "id": target_category_id,
            "name": new_class_name,
            # "supercategory": "none"
        }
    ]

    # 4. 遍历并修改所有 annotations 的 category_id
    print(f"正在转换 {len(data['annotations'])} 条标注信息...")
    for ann in data['annotations']:
        ann['category_real_id'] = ann['category_id']
        ann['category'] = target_category_id
        ann['category_id'] = target_category_id

    # 5. 保存新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"转换完成！结果已保存至: {output_json_path}")

# 使用示例
if __name__ == "__main__":
    # 假设你的原始文件名为 input.json
    # 执行后会生成 single_class_coco.json
    convert_coco_to_single_class("./datasets/proj/annotations/instances_test.json", 
                                 "./datasets/proj/annotations_clsag/instances_test_clsag.json")