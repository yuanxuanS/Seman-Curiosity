import torch
import torchvision
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
import argparse

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA is NOT available. Please check your PyTorch installation and drivers.")
        exit()

    parser = argparse.ArgumentParser("predictor demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--model", type=str, required=True, help="path to model file")
    parser.add_argument("--image", type=str, required=True, help="path to image file")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output file")
    args = parser.parse_args()
    
    # 2. 配置模型
    cfg = get_cfg()
    # 新变量直接添加
    cfg.VIS = False     
    cfg.SAVE_PTH = ''
    cfg.DATASET_NAME = ''
    cfg.MODEL.NUM_CLASSES  = 80
    cfg.merge_from_file(
        args.config
    )
    
    cfg.MODEL.WEIGHTS = args.model

    # 设置模型为评估模式 (不进行训练)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测阈值
    cfg.MODEL.DEVICE = "cuda"  # 强制使用GPU (你的RTX 5080)

    # 3. 创建预测器
    predictor = DefaultPredictor(cfg)

    # 4. 准备测试图像
    image_path = args.image 
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Please place an image or update the path.")
        exit()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Check if it's a valid image file.")
        exit()

    # 5. 进行推理
    print(f"\nRunning inference on {image_path}...")
    outputs = predictor(img)

    # 6. 可视化结果
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 7. 显示和保存结果
    output_image_path = args.output_dir + "_maskpred.png"
    # cv2.imshow("Detected Objects", out.get_image()[:, :, ::-1])
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
    print(f"Detection result saved to {output_image_path}")

