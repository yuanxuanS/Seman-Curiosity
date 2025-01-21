# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os


def infer(cfg, img_pth, save_dir):
    im = cv2.imread(img_pth)
    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)
    print(len(outputs["instances"]))
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    im = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)
    # im_ = v.get_image()
    # print(im_.shape)
    # im = v.get_image()[:, :, :]
    model_str = cfg.MODEL.WEIGHTS.split("/")[-1].split(".")[0]
    # save_pth = save_dir+"/"+img_pth.split("/")[-1].split(".")[-2]+"_"+model_str+"."+img_pth.split("/")[-1].split(".")[-1]
    save_pth = save_dir+"/"+img_pth.split("/")[-1]
    print(save_pth)
    cv2.imwrite(save_pth, im)

if __name__ == "__main__":
    # Create config
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_REAL_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    cfg.AdverTrain = False
    cfg.VIS = False
    cfg.OUTPUT_VISDIR = ""
    cfg.merge_from_file("/home/users/wpp/Look_Around_And_Learn/third_parties/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN_clsag.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set threshold for this model

    # img_pth = "/data1/wpp_data/ai2thor_all/JPEGImages/bathroom_FloorPlan401_physics_image_5.png"
    save_dir = "/data1/wpp_data/gibson_samples/vis_imgs_pred/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_dir = "/data1/wpp_data/gibson_samples/vis_imgs/"
    for img in os.listdir(img_dir):
        img_pth = img_dir+img
        infer(cfg, img_pth, save_dir)
        # break
