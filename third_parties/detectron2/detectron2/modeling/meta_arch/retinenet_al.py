# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm
import torch.distributed as dist

from ..anchor_generator import build_anchor_generator
from ..backbone import Backbone, build_backbone
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY
from .retinanet import RetinaNet, RetinaNetHead
__all__ = ["RetinaNetAL"]


logger = logging.getLogger(__name__)





@META_ARCH_REGISTRY.register()
class RetinaNetAL(RetinaNet):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        base_momentum=0.999,  # 增加
        quality_xi=0.6,    # 增加
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou"

            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).

            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at train time.
                Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        if len(self.backbone.output_shape()) != len(self.head_in_features):
            logger.warning("[RetinaNet] Backbone produces unused features.")

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        self.num_classes = num_classes
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9
        
        # AL
        self.base_momentum = base_momentum
        self.quality_xi = quality_xi

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = RetinaNetHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS),
            "anchor_matcher": Matcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
            "base_momentum": cfg.MODEL.RETINANET.BASE_MOMENTUM,
            "quality_xi": cfg.MODEL.RETINANET.QUALITY_XI,
        }

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        losses = super().losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
        loss_cls = losses["loss_cls"]
        loss_box_reg = losses["loss_box_reg"]
        # AL
        
        self.calculate_quality_and_update(pred_logits, pred_anchor_deltas, anchors, gt_labels, gt_boxes)
            
        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
        }

    def calculate_quality_and_update(self, cls_score, bbox_pred, anchors, gt_labels, gt_boxes):
        """
        Args:
            cls_score: Tensor (N, num_classes) 或 (N, num_classes+1)
            bbox_pred: Tensor (N, 4) 预测的 delta
            anchors: Tensor (N, 4)
            gt_labels: Tensor (N,) 正样本的标签
            gt_boxes: Tensor (N, 4) 对应的 ground truth boxes
        """
        device = cls_score.device
        
        with torch.no_grad():
            # 1. 解码 BBox
            # Detectron2 的 bbox_coder 对应 Box2BoxTransform
            _bbox_pred = self.box2box_transform.apply_deltas(bbox_pred, anchors)
            _bbox_gt = gt_boxes
            
            # 2. 筛选正样本 (Labels < num_classes)
            # Detectron2 中背景通常是 self.num_classes
            valid_inds = (gt_labels >= 0) & (gt_labels < self.num_classes)
            _labels = gt_labels[valid_inds]
            
            if valid_inds.any():
                # 3. 计算 IoU (is_aligned=True 对应一对一计算)
                # Detectron2 使用 pairwise_iou，对齐计算需手动或用特定函数
                from detectron2.structures import boxes as box_ops
                # 计算对齐 IoU: 
                iou = box_ops.matched_boxlist_iou(
                    box_ops.Boxes(_bbox_pred[valid_inds]), 
                    box_ops.Boxes(_bbox_gt[valid_inds])
                )
                
                # 4. 计算预测概率 p
                # 取对应 label 通道的 sigmoid 值
                pred_probs = torch.sigmoid(cls_score[valid_inds])
                p = pred_probs[torch.arange(len(_labels), device=device), _labels]
                
                # 5. 计算 Quality
                quality = torch.pow(p, self.quality_xi) * torch.pow(iou, 1. - self.quality_xi)
            else:
                _labels = torch.empty(0, dtype=torch.long, device=device)
                quality = torch.empty(0, device=device)

        # --- 开始计算分路质量均值并更新 Buffer ---
        with torch.no_grad():
            # 初始化统计量
            local_counts = torch.zeros(self.num_classes, device=device)
            local_qualities = torch.zeros(self.num_classes, device=device)
            
            # 散点累加 (Scatter Add) 替代 for 循环效率更高
            if _labels.numel() > 0:
                local_counts.scatter_add_(0, _labels, torch.ones_like(quality))
                local_qualities.scatter_add_(0, _labels, quality)
            
            # 分布式同步 (等同于 mmdet 的 concat_all_sum)
            # 此时我们将所有 GPU 的结果相加
            if comm.get_world_size() > 1:
                combined = torch.stack([local_counts, local_qualities])
                dist.all_reduce(combined) # 默认是 SUM
                global_counts, global_qualities = combined[0], combined[1]
            else:
                global_counts, global_qualities = local_counts, local_qualities

            # 计算平均质量， Eq.2
            avg_qualities = global_qualities / (global_counts + 1e-5)
            
            # 动量更新, Eq.2
            # 注意：Detectron2 的 Buffer 自动处理，无需手动加 self.
            self.class_quality = (self.class_momentum * self.class_quality + 
                                (1. - self.class_momentum) * avg_qualities)
            
            # 更新动量策略, Eq.3
            self.class_momentum = torch.where(
                avg_qualities > 0,
                torch.zeros_like(self.class_momentum) + self.base_momentum,
                self.class_momentum * self.base_momentum
            )

        return quality 


