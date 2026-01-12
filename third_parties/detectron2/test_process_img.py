import os
from PIL import Image

def rotate_vertical_images(folder_path):
    # 支持的图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # 计数器
    count = 0
    
    # 遍历文件夹
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                with Image.open(file_path) as img:
                    # 获取图像的宽和高
                    width, height = img.size
                    
                    # 检查是否为竖向图 (height > width)
                    if height > width:
                        print(f"正在旋转: {file_name} (尺寸: {width}x{height})")
                        
                        # 向右旋转90度 (顺时针)
                        # Image.ROTATE_270 是 PIL 中顺时针旋转90度的对应常量
                        # 或者使用 img.rotate(-90, expand=True)
                        rotated_img = img.transpose(Image.ROTATE_270)
                        
                        # 保存并替换原图
                        # 注意：使用 original 格式保存
                        rotated_img.save(file_path)
                        count += 1
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

    print(f"\n处理完成！共旋转并替换了 {count} 张图片。")

# 使用示例
if __name__ == "__main__":
    # 在这里输入你的文件夹路径
    target_folder = './datasets/proj/imgs/' 
    
    if os.path.exists(target_folder):
        rotate_vertical_images(target_folder)
    else:
        print("指定的文件夹路径不存在，请检查后重试。")