import pytorch_lightning as pl
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from torchmetrics.detection.map import MAP

from .roi_head_wrappers import MinimalPredictorWrapper
import torch
from torch import nn

import numpy as np

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

class Predictor(pl.LightningModule):
    '''
        build model
        prune model weights
    '''
    def __init__(self, cfg=None, 
                 input_format=None, 
                 load_checkpoint=True, 
                 metadata=None, 
                 ):
        super().__init__()
        
        cfg = setup_cfg(cfg)
        self.cfg = cfg.clone()
        
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        self.test_map_metric = MAP(class_metrics=True)
    
    def reinit_head(self, classes_idxs):    # TODO
        self.model.roi_heads.num_classes = len(classes_idxs)

        if isinstance(self.model.roi_heads.box_predictor, MinimalPredictorWrapper):
            self.model.roi_heads.box_predictor.reinit_head(classes_idxs)
        else:
            classes_to_keep = np.array([*classes_idxs, 80])
            cls_bias = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.cls_score.bias[classes_to_keep]
            )
            cls_weight = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.cls_score.weight[classes_to_keep]
            )

            classes_to_keep = np.array([*classes_idxs])
            mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
                np.arange(0, 4), len(classes_to_keep)
            )

            box_weight = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.bbox_pred.weight[mask]
            )
            box_bias = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.bbox_pred.bias[mask]
            )
            self.model.roi_heads.box_predictor.num_classes = len(classes_idxs)

            in_features = box_weight.shape[1]
            self.model.roi_heads.box_predictor.cls_score = nn.Linear(
                in_features, len(classes_to_keep) + 1
            )
            self.model.roi_heads.box_predictor.cls_score.bias = cls_bias
            self.model.roi_heads.box_predictor.cls_score.weight = cls_weight

            self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(
                in_features, len(classes_idxs) * 4
            )
            self.model.roi_heads.box_predictor.bbox_pred.bias = box_bias
            self.model.roi_heads.box_predictor.bbox_pred.weight = box_weight

            if hasattr(self.model.roi_heads, "mask_head"):
                classes_to_keep = np.array([*classes_idxs])
                mask_weight = torch.nn.Parameter(
                    self.model.roi_heads.mask_head.predictor.weight[classes_to_keep]
                )
                mask_bias = torch.nn.Parameter(
                    self.model.roi_heads.mask_head.predictor.bias[classes_to_keep]
                )
                self.model.roi_heads.mask_head.predictor.weight = mask_weight
                self.model.roi_heads.mask_head.predictor.bias = mask_bias
                self.model.roi_heads.mask_head.predictor.num_classes = len(classes_idxs)


    def on_test_epoch_end(self):
        pass
    
    def test_step(self):
        pass
    
    def forward(self):
        pass
    
    def infer(self):
        pass
    
    
        