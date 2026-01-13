import json
import os

def merge_coco_and_reset_ann_ids(file_list, output_path):
    if not file_list:
        print("错误: 文件列表为空")
        return

    # 1. 以第一个文件为模板
    with open(file_list[0], 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    # 初始化全局标注计数器
    # 注意：我们先处理第一个文件已有的标注，重置它们的ID
    global_ann_id = 0
    for ann in merged_data.get('annotations', []):
        ann['id'] = global_ann_id
        global_ann_id += 1

    base_cats_sorted = sorted(merged_data.get('categories', []), key=lambda x: x['id'])
    print(f"基准文件加载并重置初始ID完成: {file_list[0]}")

    # 2. 遍历合并剩余文件
    for i in range(1, len(file_list)):
        file_path = file_list[i]
        with open(file_path, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # 验证 categories 一致性
        current_cats_sorted = sorted(current_data.get('categories', []), key=lambda x: x['id'])
        if base_cats_sorted != current_cats_sorted:
            print(f"警告: 跳过文件 {file_path}，原因: categories 不一致")
            continue
        
        # 合并 images (假设图像ID已提前解决，不冲突)
        merged_data['images'].extend(current_data.get('images', []))
        
        # 合并并重置 annotations 的 ID
        for ann in current_data.get('annotations', []):
            new_ann = ann.copy()
            new_ann['id'] = global_ann_id  # 分配全局唯一且连续的ID
            merged_data['annotations'].append(new_ann)
            global_ann_id += 1
        
        print(f"已合并并重置标注ID: {file_path}")

    # 3. 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False) # 建议大数据集去掉 indent 减小文件体积
    
    print("-" * 30)
    print(f"最终标注总数 (Max ID): {global_ann_id}")
    print(f"结果已保存至: {output_path}")
# 获取目录下所有 json 文件并排序
json_dir = "/home/wpp/Seman-Curiosity/third_parties/detectron2/anno2"
json_files = sorted([os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')])
merge_coco_and_reset_ann_ids(json_files, 'final_merged_dataset3.json')