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
from torch import nn
import os


def feature_distance(cfg, img1_pth, img2_pth):
    criter = nn.CosineSimilarity(dim=1).cuda(1)
    im1 = cv2.imread(img1_pth)
    im2 = cv2.imread(img2_pth)
    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Get features
    features1 = predictor.extract_feature(im1)
    features2 = predictor.extract_feature(im2)

    distance_dict = {}
    for p in features1.keys():
        distance_dict[p] = criter(features1[p], features2[p]).sum()

    return distance_dict

    # print(len(outputs["instances"]))
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # plt.figure(figsize = (14, 10))
    # im = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)
    # im_ = v.get_image()
    # print(im_.shape)
    # im = v.get_image()[:, :, :]
    # model_str = cfg.MODEL.WEIGHTS.split("/")[-1].split(".")[0]
    # save_pth = save_dir+"/"+img_pth.split("/")[-1].split(".")[-2]+"_"+model_str+"."+img_pth.split("/")[-1].split(".")[-1]
    # print(save_pth)
    # cv2.imwrite(save_pth, im)

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
    save_dir = "./test_imgs"
    img_dir = "/data1/wpp_data/ai2thor_all/JPEGImages/"
    # for img in os.listdir(img_dir):
    #     img_pth = img_dir+img
    img1_pth = "/data1/wpp_data/ai2thor/Boots/image_106.png"
    img2_pth = "/data1/wpp_data/ai2thor/Boots/image_107.png"
    img3_pth = "/data1/wpp_data/ai2thor/Boots/image_108.png"
    #"/data1/wpp_data/ai2thor_all/JPEGImages/bathroom_FloorPlan403_physics_image_5.png"
    dis_12 = feature_distance(cfg, img1_pth, img2_pth)
    print(f"distance between 1-2: {dis_12.values()}")
    dis_23 = feature_distance(cfg, img2_pth, img3_pth)
    print(f"distance between 2-3: {dis_23.values()}")
