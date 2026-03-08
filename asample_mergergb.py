import os
import shutil

def merge_rgb_images_from_dict(env_dict, target_dir):
    """
    根据给定的字典将各路径下的rgb图合并到目标文件夹，并修改文件名中的环境ID。
    
    :param env_dict: 包含环境ID和对应路径的字典, 格式为 {id: "路径"}
    :param target_dir: 合并后图片存放的目标目录
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")
    target_data_dir = target_dir + "_data"
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)
        print(f"创建目标目录: {target_data_dir}")
        
    count = 0
    for env_id, folder_path in env_dict.items():
        # 确保环境ID为两位数字符串格式 (如 1 变为 '01')
        formatted_id = str(env_id).zfill(2)
        
        # 构建 rgb 文件夹的完整路径
        rgb_path = os.path.join(folder_path, 'rgb')
        
        if not os.path.exists(rgb_path):
            print(f"跳过: 在路径 {folder_path} 中未找到 rgb 文件夹")
            continue

        print(f"正在处理环境 ID {formatted_id}, 路径: {folder_path}...")

        # 遍历 rgb 文件夹下的所有图片
        for filename in os.listdir(rgb_path):
            if filename.endswith('.png'):
                # 替换文件名中的 env_00 为当前字典给定的 formatted_id
                # 原文件名示例: env_00_episode_000001_...
                new_filename = filename.replace('env_00', f'env_{formatted_id}')
                
                src_file = os.path.join(rgb_path, filename)
                dst_file = os.path.join(target_dir, new_filename)
                
                # 执行复制
                shutil.copy2(src_file, dst_file)
                count += 1

        # 构建 data 文件夹的完整路径
        data_path = os.path.join(folder_path, 'data')
        
        if not os.path.exists(data_path):
            print(f"跳过: 在路径 {folder_path} 中未找到 data 文件夹")
            continue


        # 遍历 rgb 文件夹下的所有图片
        for filename in os.listdir(data_path):
            if filename.endswith('.npy'):
                # 替换文件名中的 env_00 为当前字典给定的 formatted_id
                # 原文件名示例: env_00_episode_000001_...
                new_filename = filename.replace('env_00', f'env_{formatted_id}')
                
                src_file = os.path.join(data_path, filename)
                dst_file = os.path.join(target_data_dir, new_filename)
                
                # 执行复制
                shutil.copy2(src_file, dst_file)
                
    print(f"合并完成！总共处理了 {count} 张图片。")

# --- 手动给定环境 ID 和 路径 ---
# 您可以在这里根据实际情况修改字典内容
base_dir = "./outputs_asample/imgs/"
env_config = {
    0: base_dir + "/test5_env1",
    1: base_dir + "/test5_env2",
    2: base_dir + "/test5_env3",
    3: base_dir + "/test5_env4",
    4: base_dir + "/test5_env5"
}

# 目标输出目录
target_directory = base_dir + './test5_env1/rgb_all'

# 调用函数
merge_rgb_images_from_dict(env_config, target_directory)