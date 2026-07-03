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

from src.constants import coco_categories_mapping
from src.vqf_constants import target_coco_categories_mapping


class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.args = args

    def get_prediction(self, img, return_instance=False, return_features=False):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        if return_features:
            seg_predictions, vis_output, features = self.segmentation_model.get_predictions(
                image_list, visualize=args.visualize == 2, return_features=True)
        else:
            seg_predictions, vis_output = self.segmentation_model.get_predictions(
                image_list, visualize=args.visualize == 2, )

        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros((img.shape[0], img.shape[1], 5 + 1))

        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):
            if class_idx in list(target_coco_categories_mapping.keys()):
                idx = target_coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]['instances'].pred_masks[j] * 1.
                semantic_input[:, :, idx] += obj_mask.cpu().numpy()
                
        if not return_instance:
            if not return_features:
                return semantic_input, img
            else:
                return semantic_input, img, features
        elif return_instance:
            if not return_features:
                return semantic_input, img, seg_predictions[0]['instances']
            else:
                return semantic_input, img, seg_predictions[0]['instances'], features

    def get_predictions_batch(self, imgs, return_instance=False, return_features=False):
        """Run Mask-RCNN on a list of RGB images and return per-image outputs."""
        if len(imgs) == 0:
            if return_instance:
                return [], [], []
            return [], []
        args = self.args
        image_list = [img[:, :, ::-1] for img in imgs]
        if return_features:
            seg_predictions, vis_output, features = self.segmentation_model.get_predictions(
                image_list, visualize=False, return_features=True)
        else:
            seg_predictions, vis_output = self.segmentation_model.get_predictions(
                image_list, visualize=False)

        semantic_inputs = []
        vis_images = []
        instances = []
        for img_bgr, prediction in zip(image_list, seg_predictions):
            semantic_input = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 5 + 1))
            pred_instances = prediction['instances']
            for j, class_idx in enumerate(pred_instances.pred_classes.cpu().numpy()):
                if class_idx in list(target_coco_categories_mapping.keys()):
                    idx = target_coco_categories_mapping[class_idx]
                    obj_mask = pred_instances.pred_masks[j] * 1.
                    semantic_input[:, :, idx] += obj_mask.cpu().numpy()
            semantic_inputs.append(semantic_input)
            vis_images.append(img_bgr)
            instances.append(pred_instances)

        if return_instance:
            if return_features:
                return semantic_inputs, vis_images, instances, features
            return semantic_inputs, vis_images, instances
        if return_features:
            return semantic_inputs, vis_images, features
        return semantic_inputs, vis_images

def compress_sem_map(sem_map):
    """
    Compresses a semantic map into a single channel map by assigning each class to a unique integer.
    
    Args:
        sem_map (np.ndarray): 3D semantic map with shape (num_classes, height, width)
    
    Returns:
        np.ndarray: 2D compressed map where each pixel value represents the class index + 1
    """
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class ImageSegmentation():
    def __init__(self, args):
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

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0, return_features=False):
        return self.demo.run_on_image(img, visualize=visualize, return_features=return_features)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
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

    def run_on_image(self, image_list, visualize=0, return_features=False):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        if return_features:
            all_predictions, all_features = self.predictor(image_list, True)
        else:
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
                    instances = self.get_specific_instance(instances)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        if return_features:
            return all_predictions, vis_output, all_features
        else:
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

    def __call__(self, image_list, return_features=False):
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
            
            if return_features:
                predictions, features = self.model(inputs, return_features)
                return predictions, features
            predictions = self.model(inputs)
            return predictions
