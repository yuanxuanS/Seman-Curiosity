# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T

from .detect_utils import box_iou_calc
from src.constants import coco_categories_mapping
import cv2

class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.seg_instances = None
        self.objectness_model = ImageSegmentation(args, mode="objectness")
        self.obns_instances = None
        self.args = args

    def get_prediction(self, img):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2, specify_cls=True)
        self.seg_instances = seg_predictions
        
        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros((img.shape[0], img.shape[1], 5 + 1))

        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):
            if class_idx in list(coco_categories_mapping.keys()):
                idx = coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]['instances'].pred_masks[j] * 1.
                semantic_input[:, :, idx] += obj_mask.cpu().numpy()

        return semantic_input, img
    
    def _get_objectness_prediction(self, img):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        obns_predictions, vis_output = self.objectness_model.get_predictions(
            image_list, visualize=args.visualize == 2)
        
        self.obns_instances = obns_predictions
        
        if args.visualize == 2:
            img = vis_output.get_image()
            
        return img
    
    def get_patch_from_depth(self, depth, boxes):
        '''
        depth: (w, h)
        boxes: array, [x0, y0, x1, y1] 
        '''
        x0, y0, x1, y1 = boxes
        return depth[int(y0):int(y1), int(x0):int(x1)]
    
    def get_potential_mask(self, depth):
        
        assert self.obns_instances is not None, "call _get_objectness_prediction() first"
        assert self.seg_instances is not None
        
        device = None
        
        self.obns_instances[0]['instances'] = self.obns_instances[0]['instances'].to("cpu")
        
        
        
        width, height = self.obns_instances[0]['instances'].image_size
        pot_mp = np.zeros((height, width, 1))
        v = Visualizer(pot_mp)
        assert self.obns_instances[0]['instances'].has("pred_boxes"), "no boxes in predictions!"
        objectness_boxes = self.obns_instances[0]['instances'].pred_boxes
        
        # no objectness prediction
        if not len(objectness_boxes):  
            # pot_mp = cv2.resize(pot_mp, (self.args.frame_height, self.args.frame_width))[..., np.newaxis]   # TODO
            return pot_mp
            
        for j in range(len(objectness_boxes)):
            boxes_ = v._convert_boxes(objectness_boxes[j]).reshape(4,) # convert from 1*4 to 4*1
            
            # remove close boxes
            depth_patch = self.get_patch_from_depth(depth, boxes_)
            if depth_patch.max() < 0.5:  # 
                continue
                
            # remove box that detected by maskrcnn as well
            maskrcnn_boxes = self.seg_instances[0]['instances'].pred_boxes
            if len(maskrcnn_boxes):
                # recurse every maskrcnn's boxes to filter IoU > thes:
                device = self.seg_instances[0]['instances'].pred_boxes.device
                self.seg_instances[0]['instances'] = self.seg_instances[0]['instances'].to("cpu")
                maskrcnn_boxes = self.seg_instances[0]['instances'].pred_boxes
                
                obns_ = v._convert_boxes(objectness_boxes[j])
                msk_ = v._convert_boxes(maskrcnn_boxes)     # all maskrcnn box 
                iou = box_iou_calc(obns_, msk_) # 1*num_maskbox
                if (iou > 0.5).any():   # detected by maskrcnn as well, remove it
                    continue
            print("has far object")
            pot_mp = v.draw_patch(box_coord=boxes_, color='white')
        
        
        # resize to 128*128
        if isinstance(pot_mp, np.ndarray):
            pot_mp = pot_mp.squeeze(-1) if len(pot_mp.shape) == 3 else pot_mp
        else:
            pot_mp = pot_mp.get_image()
        
        pot_mp[pot_mp > 0] = 1.
        pot_mp= (pot_mp.astype('float32')*depth)[..., np.newaxis]    # multiply depth
        
        # cv2.imwrite(f"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/t_potential_d.png", pot_mp.transpose(1,2,0))
                
        if device is not None:
            self.seg_instances[0]['instances'] = self.seg_instances[0]['instances'].to(device)
        return pot_mp


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class ImageSegmentation():
    def __init__(self, args, mode="maskrcnn"):
        
        if mode == "objectness":
            string_args = """
                --config-file configs/quick_schedules/faster_rcnn_R_101_FPN_inference_acc_test.yaml
                --input input1.jpeg
                --confidence-threshold {}
                --opts
                """.format(args.sem_pred_prob_thr)
        else:
            string_args = """
                --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
                --input input1.jpeg
                --confidence-threshold {}
                --opts MODEL.WEIGHTS
                detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
                """.format(args.sem_pred_prob_thr)

        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))
        
        cfg = setup_cfg(args, mode)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0, specify_cls=False):
        return self.demo.run_on_image(img, visualize=visualize, specify_cls=specify_cls)

def add_new_keys(cfg):
    # 添加新key
    if not "NUM_REAL_CLASSES" in cfg.MODEL.ROI_HEADS.keys():
        cfg.MODEL.ROI_HEADS.NUM_REAL_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    if not "AdverTrain" in cfg.keys():
        cfg.AdverTrain = False
    if not "VIS" in cfg.keys():
        cfg.VIS = False
    if not "OUTPUT_VISDIR" in cfg.keys():
        cfg.OUTPUT_VISDIR = ""
    if not "BBSense_CLASSES" in cfg.keys():
        cfg.BBSense_CLASSES = ""
    return cfg
    
def setup_cfg(args, mode):
    # load config from file and command-line arguments
    cfg = get_cfg()
    
    if mode == "objectness":
        cfg = add_new_keys(cfg)
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0, specify_cls=False):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    if specify_cls:
                        instances = self.get_specific_instance(instances)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return all_predictions, vis_output

    def get_specific_instance(self, instances):
        if len(instances) == 0:
            return instances
        instance_lst = []
        for i, cls_id in enumerate(instances.pred_classes):
            if cls_id.cpu().numpy() in list(coco_categories_mapping.keys()):
                instance_lst.append(instances[i])
        if len(instance_lst) == 0:
            return Instances(instances[0]._image_size)
        instance_res = instances.cat(instance_lst)
        return instance_res
        

class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
