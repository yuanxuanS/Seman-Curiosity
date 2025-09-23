import os
import shutil

def rename_and_move_files(source_dir, target_dir, env_num):
    """
    重命名文件并将它们移动到目标文件夹
    
    参数:
        source_dir: 源文件夹路径(包含原始文件)
        target_dir: 目标文件夹路径
        env_num: 要替换的环境编号(两位数，如1要写成01)
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    cnt = 0
    for filename in os.listdir(source_dir):
        if filename.startswith("env_00") and filename.endswith(".npy"):
            # 分割文件名
            parts = filename.split("_")
            
            # # 替换环境编号
            parts[1] = f"{env_num:02d}"  # 格式化为两位数
            
            # 重新组合文件名
            new_filename = "_".join(parts)
            
            # 源文件完整路径
            src_path = os.path.join(source_dir, filename)
            # 目标文件完整路径
            dst_path = os.path.join(target_dir, new_filename)
            
            # 移动并重命名文件
            shutil.move(src_path, dst_path)
            print(f"Moved and renamed: {filename} -> {new_filename}")

            cnt += 1
    print(f"conut:", cnt)
# 使用示例
if __name__ == "__main__":
    source_folder = "/home/wpp/Seman-Curiosity/exps/dump/vsqf_v4_eval_ex/episodes_data"
    # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/random4vsqf_ex/episodes_data"  # 替换为你的源文件夹路径
    target_folder = "/home/wpp/Seman-Curiosity/exps/dump/vsqf_v4_eval/episodes_data"  # 替换为目标文件夹路径
    new_env_number = 4  # 要替换为的环境编号
    
    rename_and_move_files(source_folder, target_folder, new_env_number)
