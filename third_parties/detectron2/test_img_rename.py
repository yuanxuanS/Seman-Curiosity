import json
import os

def rename_images_and_update_coco(json_path, image_dir, output_json_path):
    # 1. 加载标注文件
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 遍历图像信息进行处理
    for img_node in coco_data['images']:
        img_id = img_node['id']
        old_name = img_node['file_name']
        
        # 新的文件名：ID_原名
        new_name = f"{img_id}_{old_name}"
        
        old_path = os.path.join(image_dir, old_name)
        new_path = os.path.join(image_dir, new_name)

        # 3. 执行物理重命名 (检查文件是否存在防止报错)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"成功: {old_name} -> {new_name}")
        else:
            print(f"警告: 未找到文件 {old_path}")

        # 4. 修改内存中的 JSON 数据
        img_node['file_name'] = new_name
# 
    # 5. 保存修改后的标注文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n全部完成！新的标注文件已保存至: {output_json_path}")

# --- 配置参数 ---
json_file = "./datasets/proj/annotations/instances_test3.json"  # 你的原始 JSON 文件路径
img_folder = "./datasets/proj/test"           # 存放图像的文件夹路径
output_json = "./datasets/proj/annotations/instances_test.json" # 修改后的 JSON 保存路径

# 执行
rename_images_and_update_coco(json_file, img_folder, output_json)