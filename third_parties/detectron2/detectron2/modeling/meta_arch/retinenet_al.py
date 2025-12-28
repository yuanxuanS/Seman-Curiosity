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
from detectron2.structures import boxes as box_ops
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K

from ..anchor_generator import build_anchor_generator
from ..backbone import Backbone, build_backbone
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY
from .retinanet import RetinaNet, RetinaNetHead
from detectron2.utils.registry import Registry

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")

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
        infer=False,    # 增加
        infer_diversity=False,
        backbone_outshape=None,
        max_det=100,
        feat_dim=256,
        total_images=0,
        output_path="",
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
        super().__init__(
            backbone=backbone,
            head=head,
            head_in_features=head_in_features,
            anchor_generator=anchor_generator,
            box2box_transform=box2box_transform,
            anchor_matcher=anchor_matcher,
            num_classes=num_classes,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            test_score_thresh=test_score_thresh,
            test_topk_candidates=test_topk_candidates,
            test_nms_thresh=test_nms_thresh,
            max_detections_per_image=max_detections_per_image,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            vis_period=vis_period,
            input_format=input_format,
            
        )

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
        self.register_buffer('class_momentum', torch.ones((num_classes,)) * base_momentum)
        self.register_buffer('class_quality', torch.zeros((num_classes,)))
        self.infer = infer
        print(f"infer:  in retina{infer}")
        
        self.infer_diversity = infer_diversity
        # 新增：记录当前队列指针位置（如果是跨 batch 积累，这很重要）
        self.current_images = 0
        self.max_det = max_det
        self.feat_dim = feat_dim
        
        world_size = comm.get_world_size()
        assert total_images % world_size == 0  # 8 GPUs
        self.total_images = total_images
        self.queue_length = total_images
        if self.infer_diversity:
            # 类别标签通常为 long 类型，用于索引
            self.register_buffer("det_label_queue", torch.zeros((self.queue_length, max_det), dtype=torch.long))

            # 置信度分数和特征使用 float32 (默认)
            self.register_buffer("det_score_queue", torch.zeros((self.queue_length, max_det), dtype=torch.float32))
            self.register_buffer("det_feat_queue", torch.zeros((self.queue_length, max_det, feat_dim), dtype=torch.float32))

            # 图像 ID 使用 int 类型，初始化为 -1
            self.register_buffer("image_id_queue", torch.full((self.queue_length, 1), -1, dtype=torch.int32))

            self.output_path = output_path
            
            # 手动获取每层的 stride 并转换为 scale
            # input_shape 是由 backbone 提供的各层信息
            pooler_scales = tuple(1.0 / backbone_outshape[k].stride for k in self.head_in_features)
            
            # 定义 Pooler
            # 如果只是为了提取特征向量，output_size 可以设为 1 或 7
            from detectron2.modeling.poolers import ROIPooler
            self.box_pooler = ROIPooler(
                output_size=1,            # 如果需要 1x1 的特征向量
                scales=pooler_scales,
                sampling_ratio=0,         # 0 代表自动计算
                pooler_type="ROIAlignV2", # 推荐使用 ROIAlignV2
            )
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        
        
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        if hasattr(cfg.MODEL, "HEAD"):
            if cfg.MODEL.HEAD == 'RetinaHeadFeat':
                head = RetinaHeadFeat(cfg, feature_shapes)
            else:
                head = RetinaNetHead(cfg, feature_shapes)
        else:
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
            "infer": cfg.INFER,
            "infer_diversity": cfg.INFER_DIVERSITY,
            "backbone_outshape": backbone_shape,
            "max_det": cfg.MAX_DET if hasattr(cfg, "MAX_DET") else 100,
            "feat_dim": cfg.FEAT_DIM if hasattr(cfg, "FEAT_DIM") else 256,
            "total_images": cfg.TOTAL_IMAGES if hasattr(cfg, "TOTAL_IMAGES") else 0,
            "output_path": cfg.OUTPUT_PATH if hasattr(cfg, "OUTPUT_PATH") else "",
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
                N =batch size
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


    def calculate_quality_and_update(self, pred_logits, pred_anchor_deltas, anchors, gt_labels, gt_boxes):
        """
        Args:
            pred_logits: list[Tensor], 每层形状 (N, R_i, K)
            pred_anchor_deltas: list[Tensor], 每层形状 (N, R_i, 4)
            anchors: list[Boxes], 每层长度为 R_i 的 Boxes (单张图的 anchor)
            gt_labels: list[Tensor], 每个 Batch 形状 (R,)，R = sum(R_i)
            gt_boxes: list[Tensor], 每个 Batch 形状 (R, 4)
        """
        device = pred_logits[0].device
        num_imgs = len(gt_labels) # N

        # 1. 展平并合并预测值 (N, R, K) -> (N*R, K)
        # R 是所有 feature levels 的 anchor 总数
        flat_logits = torch.cat(pred_logits, dim=1).reshape(-1, self.num_classes)
        flat_deltas = torch.cat(pred_anchor_deltas, dim=1).reshape(-1, 4)

        # 2. 展平并合并 Anchors (R, 4) -> (N*R, 4)
        # anchors 只有一份（单图），需要根据 Batch Size 复制
        single_img_anchors = torch.cat([b.tensor for b in anchors], dim=0) # (R, 4)
        flat_anchors = single_img_anchors.repeat(num_imgs, 1) # (N*R, 4)

        # 3. 展平并合并 GT (list[Tensor]) -> (N*R, ...)
        flat_gt_labels = torch.cat(gt_labels, dim=0) # (N*R,)
        flat_gt_boxes = torch.cat(gt_boxes, dim=0)   # (N*R, 4)

        with torch.no_grad():
            # 4. 解码所有预测框 (N*R, 4)
            # 使用 D2 的 box2box_transform
            _bbox_pred = self.box2box_transform.apply_deltas(flat_deltas, flat_anchors)
            
            # 5. 筛选正样本索引 (Valid Indices)
            # D2 中正样本通常为 0 ~ num_classes-1
            valid_inds = (flat_gt_labels >= 0) & (flat_gt_labels < self.num_classes)
            _labels = flat_gt_labels[valid_inds]
            
            if valid_inds.any():
                # 6. 计算对齐 IoU (一对一计算每个正样本 Anchor 对应的预测与 GT)
                # matched_boxlist_iou 输入要求是 Boxes 对象
                iou = box_ops.matched_boxlist_iou(
                    Boxes(_bbox_pred[valid_inds]), 
                    Boxes(flat_gt_boxes[valid_inds])
                )
                
                # 7. 计算分类预测概率 p
                # RetinaNet 使用 Sigmoid。我们提取对应正确类别的概率值
                pred_probs = torch.sigmoid(flat_logits[valid_inds])
                # 使用 arange 索引提取每个样本对应其标签通道的得分
                p = pred_probs[torch.arange(len(_labels), device=device), _labels]
                
                # 8. 计算质量分数 Quality (Eq.1),  difficulty = 1- quality, 在DCUS sampler中转化为difficulty
                quality = torch.pow(p, self.quality_xi) * torch.pow(iou, 1. - self.quality_xi)
            else:
                _labels = torch.empty(0, dtype=torch.long, device=device)
                quality = torch.empty(0, device=device)

        # --- 统计更新与分布式同步 ---
        with torch.no_grad():
            # 初始化当前卡上的累加器
            local_counts = torch.zeros(self.num_classes, device=device)
            local_qualities = torch.zeros(self.num_classes, device=device)
            
            if _labels.numel() > 0:
                # 使用 scatter_add 将不同类别的 quality 分别累加
                local_counts.scatter_add_(0, _labels, torch.ones_like(quality))
                local_qualities.scatter_add_(0, _labels, quality)
            
            # 分布式全归约 (SUM)，同步所有 GPU 的统计结果
            if comm.get_world_size() > 1:
                combined = torch.stack([local_counts, local_qualities])
                dist.all_reduce(combined, op=dist.ReduceOp.SUM)
                global_counts, global_qualities = combined[0], combined[1]
            else:
                global_counts, global_qualities = local_counts, local_qualities

            # 9. 计算全局类别平均质量 (Eq.2)
            avg_qualities = global_qualities / (global_counts + 1e-5)
            
            # 10. 动量更新 Buffer (Eq.2)
            # self.class_quality 应在 __init__ 中通过 register_buffer 注册
            self.class_quality = (self.class_momentum * self.class_quality + 
                                (1. - self.class_momentum) * avg_qualities)
            
            # 11. 更新动量衰减策略 (Eq.3)
            # 如果某类别在当前 batch 没出现，动量会随着 base_momentum 逐渐衰减
            self.class_momentum = torch.where(
                avg_qualities > 0,
                torch.full_like(self.class_momentum, self.base_momentum),
                self.class_momentum * self.base_momentum
            )

        return quality
    
    
    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        if self.infer_diversity:
            all_level_indices = [] # 新增：记录层级
        
        # Iterate over every feature level
        for i, (box_cls_i, box_reg_i, anchors_i) in enumerate(zip(box_cls, box_delta, anchors)):
        # for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            
            if self.infer_diversity:
                # 为当前层级保留下来的每一个框都标记上索引 i
                level_i = torch.full_like(classes_idxs, i, dtype=torch.long)
                all_level_indices.append(level_i)
        
        if self.infer_diversity:
            boxes_all, scores_all, class_idxs_all, all_level_indices = [
                cat(x) for x in [boxes_all, scores_all, class_idxs_all, all_level_indices]
            ]
        else:
            boxes_all, scores_all, class_idxs_all = [
                cat(x) for x in [boxes_all, scores_all, class_idxs_all]
            ]
            
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        if self.infer:
            # --- 新增：计算分类不确定性 (Uncertainty) ---
            # 提取 NMS 后的最终分数
            final_scores = scores_all[keep]
            
            # 使用信息熵公式: H = -[p*log(p) + (1-p)*log(1-p)]
            eps = 1e-10
            cls_uncertainties = -1 * (
                final_scores * torch.log(final_scores + eps) + 
                (1 - final_scores) * torch.log((1 - final_scores) + eps)
            )
            # 初始化 box_uncertainty 为 0 (MMDet 代码中也是如此)
            box_uncertainties = torch.zeros_like(cls_uncertainties)
        
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        
        if self.infer:
            # 将计算的不确定性存入 Instances 对象中，以便 Evaluator 或可视化使用
            result.cls_uncertainty = cls_uncertainties
            result.box_uncertainty = box_uncertainties
            
        if self.infer_diversity:
            result.level_indices = all_level_indices[keep]
        return result
    
    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        

        anchors = self.anchor_generator(features)
        if self.infer_diversity:
            pred_logits, pred_anchor_deltas, cls_features = self.head(features)
        else:
            pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for i, (results_per_image, input_per_image, image_size) in enumerate(zip(
                results, batched_inputs, images.image_sizes
            )):  
                if self.infer_diversity:
                    # 2. 计算分类不确定性 (Shannon Entropy) 
                    # 对应  cls_uncertainties = -1 * (p*log(p) + (1-p)*log(1-p))
                    scores = results_per_image.scores
                    eps = 1e-10
                    # 计算信息熵作为不确定性指标
                    cls_uncertainties = -1 * (scores * torch.log(scores + eps) + (1 - scores) * torch.log(1 - scores + eps))
                    
                    # 存入结果实例中
                    results_per_image.cls_uncertainties = cls_uncertainties
                    results_per_image.box_uncertainties = torch.zeros_like(cls_uncertainties)

                    # 3. 特征收集逻辑 (对应 MMDet 的 get_inter_feats)
                    # 提取 ROI 特征，对应你 MMDet 代码里的 det_feats
                    # 我们需要重新包装单张图的 feature dict
                    curr_img_feat = [feat[i][None, ...] for feat in cls_features] # 这里可以根据需要调整索引
                    det_feats = self.get_inter_feats_d2_exact(curr_img_feat, results_per_image, image_size)
                    results_per_image.det_feats = det_feats

                    # 4. 收集信息并处理队列 (对应 MMDet 的 collect_det_info)
                    # 你可以在这里调用自定义的汇总函数
                    rank, world_size = self.collect_al_info_d2(input_per_image, results_per_image)
                    self.current_images += world_size
                    
                    if self.current_images >= self.total_images:
                        torch.cuda.empty_cache()
                        if rank == 0:
                            self.compute_al()
                        else:
                            torch.cuda.synchronize()
                            
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def get_inter_feats_d2_exact(self, lvl_feats, pred_instances, img_shape):
        """
        完全对应 MMDetection 的 get_inter_feats 逻辑。
        
        Args:
            lvl_feats (list[Tensor]): 对应 curr_img_feat，每层形状为 [1, C, H, W]
            pred_instances (Instances): 包含 pred_boxes, scores 
                                        以及自定义的 level_indices (关键！)
            img_shape (tuple): (height, width)
        """
        device = lvl_feats[0].device
        num_det = len(pred_instances)
        channels = lvl_feats[0].shape[1]
        img_h, img_w = img_shape
        
        # 获取框的中心点
        # boxes 形状为 [num_det, 4] (x1, y1, x2, y2)
        boxes = pred_instances.pred_boxes.tensor
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5

        # 归一化坐标到 [-1, 1] 范围，适配 F.grid_sample
        # 逻辑：((coord / size) - 0.5) * 2 => (coord / size) * 2 - 1
        nx = (cx / img_w) * 2 - 1
        ny = (cy / img_h) * 2 - 1
        
        coor = torch.stack((nx, ny), dim=-1) # [num_det, 2]
        ret_feats = torch.zeros((num_det, channels), device=device)

        # 必须有 level_indices，这代表每个预测框是从哪一层 FPN 出来的
        if not pred_instances.has("level_indices"):
            raise AttributeError("Instances must have 'level_indices'. Please modify inference_single_image to include it.")
        
        lvl_inds = pred_instances.level_indices

        for l in range(len(lvl_feats)):
            mask_l = (lvl_inds == l)
            if not mask_l.any():
                continue

            feat_l = lvl_feats[l]  # 已经是 [1, C, H, W]
            
            # 构造 grid_sample 要求的 grid 形状: [1, 1, n_det_lvl, 2]
            # grid 内容是 (x, y) 且范围在 [-1, 1]
            coor_l = coor[mask_l].view(1, 1, -1, 2)
            
            # 进行双线性插值采样
            # inter_feat 形状: [1, C, 1, n_det_lvl]
            inter_feat = F.grid_sample(feat_l, coor_l, mode='bilinear', padding_mode='zeros', align_corners=False)
            
            # 转换形状回 [n_det_lvl, C] 并填入结果
            inter_feat = inter_feat.squeeze(0).squeeze(1).transpose(0, 1)
            ret_feats[mask_l] = inter_feat

        return ret_feats

    def collect_al_info_d2(self, img_meta, instances):
        """
        Args:
            img_meta (dict): 从 batched_inputs 提取的单图信息，包含 'file_name'。
            instances (Instances): 包含 scores, pred_classes, det_feats 的结果对象。
        """
        def concat_all_gather(tensor):
            """
            在所有进程上收集 tensor 并将其拼接。
            """
            tensors_gather = comm.all_gather(tensor)
            return torch.cat(tensors_gather, dim=0)

        # 1. 获取分布式信息
        rank = comm.get_rank()
        world_size = comm.get_world_size()

        # 2. 提取并准备图片 ID (与 MMDet 逻辑一致)
        # D2 中通常是 'file_name' 对应 MMDet 的 'filename'
        file_path = img_meta["file_name"]
        img_id_val = int(file_path.split('/')[-1].split('.')[0])
        img_id_tensor = torch.tensor([[img_id_val]], dtype=torch.int, device=self.device)

        # 3. 准备数据进行收集
        # 注意：Instances 中的数据需要对齐到固定长度 max_det (padding)
        # 以防 NMS 后的框数量少于要求的 max_det
        max_det = self.max_det  # 你在 __init__ 中定义的预设值
        num_actual = len(instances)
        
        # 对 Labels, Scores, Feats 进行 Padding 处理
        det_labels = torch.zeros(max_det, dtype=torch.long, device=self.device)
        det_scores = torch.zeros(max_det, device=self.device)
        det_feats = torch.zeros((max_det, self.feat_dim), device=self.device)

        actual_num = min(num_actual, max_det)
        if actual_num > 0:
            det_labels[:actual_num] = instances.pred_classes[:actual_num]
            det_scores[:actual_num] = instances.scores[:actual_num]
            det_feats[:actual_num] = instances.det_feats[:actual_num]

        # 4. 执行分布式收集 (All Gather)
        # 对齐维度以满足 concat_all_gather 要求 [1, ...]
        collected_img_ids = concat_all_gather(img_id_tensor.reshape(1, 1))
        collected_det_labels = concat_all_gather(det_labels.reshape(1, max_det))
        collected_det_scores = concat_all_gather(det_scores.reshape(1, max_det).contiguous())
        collected_det_feats = concat_all_gather(det_feats.reshape(1, max_det, self.feat_dim).contiguous())

        # 5. 更新本地队列
        # self.current_images 需在 forward 的循环中外部维护或作为类属性
        start_idx = self.current_images
        end_idx = self.current_images + world_size
        
        # 确保索引不溢出（取决于你的队列大小设计）
        self.image_id_queue[start_idx:end_idx] = collected_img_ids
        self.det_label_queue[start_idx:end_idx] = collected_det_labels
        self.det_score_queue[start_idx:end_idx] = collected_det_scores
        self.det_feat_queue[start_idx:end_idx] = collected_det_feats
        return rank, world_size

    def compute_al(self):
        
        def get_img_score_distance_matrix_slow(
            all_labels,
            all_scores,
            all_feats,
            score_thr=0.,
            same_label=True,
            metric='cosine'):

            INF = 1e12
            assert metric in ('l2', 'cosine', 'kl')

            n_images = all_labels.size(0)
            n_dets = all_labels.size(1)
            feat_dim = all_feats.size(-1)
            dets_indices = torch.arange(n_dets).to(device=all_feats.device)

            if metric == 'cosine':
                all_feats = F.normalize(all_feats, p=2, dim=-1)
                all_feats_t = all_feats.transpose(1, 2)

                all_score_valid = (all_scores > score_thr).to(dtype=all_feats.dtype)
                all_score_valid_t = all_score_valid[:, :, None].transpose(1, 2)

                all_scores_t = all_scores[:, :, None].transpose(1, 2)
                all_labels_t = all_labels[:, :, None].transpose(1, 2)

                distances = []
                for i in range(n_images):
                    # torch.cuda.empty_cache()

                    labels_i = all_labels[i]  # [n_dets]
                    scores_valid_i = all_score_valid[i]  # [n_dets]
                    scores_i = all_scores[i]  # [n_dets]
                    feats_i = all_feats[i]  # [n_dets, feat_dim]
                    
                    # 
                    feat_distances_i =  -1 * torch.matmul(feats_i.view(1, n_dets, feat_dim), all_feats_t) + 1 # [n_images, n_dets, n_dets]
                    feat_distances_i[:,dets_indices, dets_indices] = 0  # force diag to 0, avoid numerical unstable
                    score_valid = torch.matmul(scores_valid_i.view(1, n_dets, 1), all_score_valid_t)  # [n_images, n_dets, n_dets]

                    if same_label:
                        labels_i = labels_i[:, None].repeat(1,n_dets) # [n_dets, n_dets]
                        label_valid = (labels_i.view(1, n_dets, n_dets) == all_labels_t).to(dtype=all_feats.dtype)
                    else:
                        label_valid = torch.ones_like(score_valid)

                    label_invalid = (1 - label_valid).to(dtype=torch.bool)
                    score_invalid = (1 - score_valid).to(dtype=torch.bool)

                    feat_distances_i[label_invalid] = 2.
                    feat_distances_i[score_invalid] = INF

                    feat_distances_i = feat_distances_i.min(dim=-1)[0]  # [n_images, n_dets]

                    norm = (score_valid.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
                    '''
                    Potential BUG: 
                    If no box > score_thr in both images, the algorithm fails. But this is unlikely to happen
                    '''
                    feat_distances_i[feat_distances_i > 2] = 0.

                    # Eq. 7
                    feat_distances_i = feat_distances_i * scores_i[None, :]
                    feat_distances_i = feat_distances_i.sum(dim=-1) / norm
                    distances.append(feat_distances_i.cpu())

                feat_distance = torch.stack(distances, dim=0)
                feat_distance = 0.5 * (feat_distance + feat_distance.transpose(0, 1))
                return feat_distance

            elif metric == 'kl':
                assert not same_label

                all_score_valid = (all_scores > score_thr).to(dtype=all_feats.dtype)
                all_score_valid_t = all_score_valid[:, :, None].transpose(1, 2)

                all_scores_t = all_scores[:, :, None].transpose(1, 2)
                all_labels_t = all_labels[:, :, None].transpose(1, 2)

                distances = []
                for i in range(n_images):

                    labels_i = all_labels[i]  # [n_dets]
                    scores_valid_i = all_score_valid[i]  # [n_dets]
                    scores_i = all_scores[i]  # [n_dets]
                    feats_i = all_feats[i]  # [n_dets, feat_dim]
                    feat_distances_i = []
                    eps = 1e-12
                    _pred = feats_i.view(1, n_dets, 1, feat_dim).repeat(1, 1, n_dets, 1)
                    band_width = 20
                    assert n_images % band_width == 0
                    for j in range(n_images // band_width):
                        _target = all_feats[j*band_width:(j+1)*band_width].view(band_width,1, n_dets, feat_dim).repeat(1,n_dets,1,1)
                        kl = _target * ((_target+eps).log() - (_pred+eps).log())
                        feat_distances_i.append(kl.sum(dim=-1))
                    feat_distances_i =  torch.cat(feat_distances_i, dim = 0)
                    feat_distances_i[:, dets_indices, dets_indices] = 0  # force diag to 0, avoid numerical unstable
                    score_valid = torch.matmul(scores_valid_i.view(1, n_dets, 1),
                                            all_score_valid_t)  # [n_images, n_dets, n_dets]

                    if same_label:
                        labels_i = labels_i[:, None].repeat(1, n_dets)  # [n_dets, n_dets]
                        label_valid = (labels_i.view(1, n_dets, n_dets) == all_labels_t).to(dtype=all_feats.dtype)
                    else:
                        label_valid = torch.ones_like(score_valid)

                    label_invalid = (1 - label_valid).to(dtype=torch.bool)
                    score_invalid = (1 - score_valid).to(dtype=torch.bool)

                    feat_distances_i[label_invalid] = 2.
                    feat_distances_i[score_invalid] = INF

                    feat_distances_i = feat_distances_i.min(dim=-1)[0]  # [n_images, n_dets]
                    norm = (score_valid.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
                    '''
                    Potential BUG: 
                    If no box > score_thr in both images, the algorithm fails. But this is unlikely to happen
                    '''
                    feat_distances_i[feat_distances_i > 2] = 0.
                    feat_distances_i = feat_distances_i * scores_i[None, :]
                    feat_distances_i = feat_distances_i.sum(dim=-1) / norm
                    distances.append(feat_distances_i.cpu())

                feat_distance = torch.stack(distances, dim=0)
                feat_distance = 0.5 * (feat_distance + feat_distance.transpose(0, 1))
                return feat_distance

            else:
                raise NotImplementedError
                return None


        valid_inds = (self.image_id_queue >= 0).reshape(-1)
        image_id_queue = self.image_id_queue[valid_inds]

        det_label_queue = self.det_label_queue[valid_inds]
        det_score_queue = self.det_score_queue[valid_inds]
        det_feat_queue = self.det_feat_queue[valid_inds]

        img_dis_mat = get_img_score_distance_matrix_slow(
            det_label_queue, det_score_queue, det_feat_queue, score_thr=0.05)

        img_dis_mat = img_dis_mat.detach().cpu().numpy()
        img_ids = image_id_queue.detach().cpu().numpy()

        with open(self.output_path, 'wb') as fwb:
            np.save(fwb, img_dis_mat)
            np.save(fwb, img_ids)
        return



@ROI_HEADS_REGISTRY.register()
class RetinaHeadFeat(RetinaNetHead):
    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            num_anchors=num_anchors,
            conv_dims=conv_dims,
            norm=norm,
            prior_prob=prior_prob,
        )

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        cls_feats = []
        for feature in features:
            cls_subnet_feat = self.cls_subnet(feature)
            cls_feats.append(cls_subnet_feat)
            
            logits.append(self.cls_score(cls_subnet_feat))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg, cls_feats