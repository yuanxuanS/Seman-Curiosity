# from .roi_heads import ROIHeads, ROI_HEADS_REGISTRY
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from ..poolers import ROIPooler
import inspect
from .box_head import build_box_head
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from .mask_head import build_mask_head
from .keypoint_head import build_keypoint_head
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform
from .fast_rcnn import _log_classification_stats, fast_rcnn_inference
from fvcore.nn import giou_loss, smooth_l1_loss

# @ROI_HEADS_REGISTRY.register()
# class AdverROIHeads(ROIHeads):
#     """
#     box predictor, object-or-not biregressor, adversarial cls pred head

#     To implement more models, you can subclass it and implement a different
#     :meth:`forward()` or a head.
#     """

#     @configurable
#     def __init__(
#         self,
#         *,
#         box_in_features: List[str],
#         box_pooler: ROIPooler,
#         box_head: nn.Module,
#         box_predictor: nn.Module = AdverOutputLayers,
#         mask_in_features: Optional[List[str]] = None,
#         mask_pooler: Optional[ROIPooler] = None,
#         mask_head: Optional[nn.Module] = None,
#         keypoint_in_features: Optional[List[str]] = None,
#         keypoint_pooler: Optional[ROIPooler] = None,
#         keypoint_head: Optional[nn.Module] = None,
#         train_on_pred_boxes: bool = False,
#         **kwargs,
#     ):
#         """
#         NOTE: this interface is experimental.

#         Args:
#             box_in_features (list[str]): list of feature names to use for the box head.
#             box_pooler (ROIPooler): pooler to extra region features for box head
#             box_head (nn.Module): transform features to make box predictions
#             box_predictor (nn.Module): make box predictions from the feature.
#                 Should have the same interface as :class:`FastRCNNOutputLayers`.
#             mask_in_features (list[str]): list of feature names to use for the mask
#                 pooler or mask head. None if not using mask head.
#             mask_pooler (ROIPooler): pooler to extract region features from image features.
#                 The mask head will then take region features to make predictions.
#                 If None, the mask head will directly take the dict of image features
#                 defined by `mask_in_features`
#             mask_head (nn.Module): transform features to make mask predictions
#             keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
#             train_on_pred_boxes (bool): whether to use proposal boxes or
#                 predicted boxes from the box head to train other heads.
#         """
#         super().__init__(**kwargs)
#         # keep self.in_features for backward compatibility
#         self.in_features = self.box_in_features = box_in_features
#         self.box_pooler = box_pooler
#         self.box_head = box_head
#         self.box_predictor = box_predictor

#         self.mask_on = mask_in_features is not None
#         if self.mask_on:
#             self.mask_in_features = mask_in_features
#             self.mask_pooler = mask_pooler
#             self.mask_head = mask_head

#         self.keypoint_on = keypoint_in_features is not None
#         if self.keypoint_on:
#             self.keypoint_in_features = keypoint_in_features
#             self.keypoint_pooler = keypoint_pooler
#             self.keypoint_head = keypoint_head

#         self.train_on_pred_boxes = train_on_pred_boxes

#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         ret = super().from_config(cfg)
#         ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
#         # Subclasses that have not been updated to use from_config style construction
#         # may have overridden _init_*_head methods. In this case, those overridden methods
#         # will not be classmethods and we need to avoid trying to call them here.
#         # We test for this with ismethod which only returns True for bound methods of cls.
#         # Such subclasses will need to handle calling their overridden _init_*_head methods.
#         if inspect.ismethod(cls._init_box_head):
#             ret.update(cls._init_box_head(cfg, input_shape))
#         if inspect.ismethod(cls._init_mask_head):
#             ret.update(cls._init_mask_head(cfg, input_shape))
#         if inspect.ismethod(cls._init_keypoint_head):
#             ret.update(cls._init_keypoint_head(cfg, input_shape))
#         return ret

#     @classmethod
#     def _init_box_head(cls, cfg, input_shape):
#         # fmt: off
#         in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
#         pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
#         pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
#         sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
#         # fmt: on

#         # If StandardROIHeads is applied on multiple feature maps (as in FPN),
#         # then we share the same predictors and therefore the channel counts must be the same
#         in_channels = [input_shape[f].channels for f in in_features]
#         # Check all channel counts are equal
#         assert len(set(in_channels)) == 1, in_channels
#         in_channels = in_channels[0]

#         box_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )
#         # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
#         # They are used together so the "box predictor" layers should be part of the "box head".
#         # New subclasses of ROIHeads do not need "box predictor"s.
#         box_head = build_box_head(
#             cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
#         )
#         box_predictor = AdverOutputLayers(cfg, box_head.output_shape)
#         return {
#             "box_in_features": in_features,
#             "box_pooler": box_pooler,
#             "box_head": box_head,
#             "box_predictor": box_predictor,
#         }

#     @classmethod
#     def _init_mask_head(cls, cfg, input_shape):
#         if not cfg.MODEL.MASK_ON:
#             return {}
#         # fmt: off
#         in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
#         pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
#         pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
#         sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
#         # fmt: on

#         in_channels = [input_shape[f].channels for f in in_features][0]

#         ret = {"mask_in_features": in_features}
#         ret["mask_pooler"] = (
#             ROIPooler(
#                 output_size=pooler_resolution,
#                 scales=pooler_scales,
#                 sampling_ratio=sampling_ratio,
#                 pooler_type=pooler_type,
#             )
#             if pooler_type
#             else None
#         )
#         if pooler_type:
#             shape = ShapeSpec(
#                 channels=in_channels, width=pooler_resolution, height=pooler_resolution
#             )
#         else:
#             shape = {f: input_shape[f] for f in in_features}
#         ret["mask_head"] = build_mask_head(cfg, shape)
#         return ret

#     @classmethod
#     def _init_keypoint_head(cls, cfg, input_shape):
#         if not cfg.MODEL.KEYPOINT_ON:
#             return {}
#         # fmt: off
#         in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
#         pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
#         pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
#         sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
#         # fmt: on

#         in_channels = [input_shape[f].channels for f in in_features][0]

#         ret = {"keypoint_in_features": in_features}
#         ret["keypoint_pooler"] = (
#             ROIPooler(
#                 output_size=pooler_resolution,
#                 scales=pooler_scales,
#                 sampling_ratio=sampling_ratio,
#                 pooler_type=pooler_type,
#             )
#             if pooler_type
#             else None
#         )
#         if pooler_type:
#             shape = ShapeSpec(
#                 channels=in_channels, width=pooler_resolution, height=pooler_resolution
#             )
#         else:
#             shape = {f: input_shape[f] for f in in_features}
#         ret["keypoint_head"] = build_keypoint_head(cfg, shape)
#         return ret

#     def forward(
#         self,
#         images: ImageList,
#         features: Dict[str, torch.Tensor],
#         proposals: List[Instances],
#         targets: Optional[List[Instances]] = None,
#     ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
#         """
#         See :class:`ROIHeads.forward`.
#         """
#         del images
#         if self.training:
#             assert targets, "'targets' argument is required during training"
#             proposals = self.label_and_sample_proposals(proposals, targets)
#         del targets

#         if self.training:
#             losses = self._forward_box(features, proposals)
#             # Usually the original proposals used by the box head are used by the mask, keypoint
#             # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
#             # predicted by the box head.
#             losses.update(self._forward_mask(features, proposals))
#             losses.update(self._forward_keypoint(features, proposals))
#             return proposals, losses
#         else:
#             pred_instances = self._forward_box(features, proposals)
#             # During inference cascaded prediction is used: the mask and keypoints heads are only
#             # applied to the top scoring box detections.
#             pred_instances = self.forward_with_given_boxes(features, pred_instances)
#             return pred_instances, {}

#     def forward_with_given_boxes(
#         self, features: Dict[str, torch.Tensor], instances: List[Instances]
#     ) -> List[Instances]:
#         """
#         Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

#         This is useful for downstream tasks where a box is known, but need to obtain
#         other attributes (outputs of other heads).
#         Test-time augmentation also uses this.

#         Args:
#             features: same as in `forward()`
#             instances (list[Instances]): instances to predict other outputs. Expect the keys
#                 "pred_boxes" and "pred_classes" to exist.

#         Returns:
#             list[Instances]:
#                 the same `Instances` objects, with extra
#                 fields such as `pred_masks` or `pred_keypoints`.
#         """
#         assert not self.training
#         assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

#         instances = self._forward_mask(features, instances)
#         instances = self._forward_keypoint(features, instances)
#         return instances

#     def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
#         """
#         Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
#             the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

#         Args:
#             features (dict[str, Tensor]): mapping from feature map names to tensor.
#                 Same as in :meth:`ROIHeads.forward`.
#             proposals (list[Instances]): the per-image object proposals with
#                 their matching ground truth.
#                 Each has fields "proposal_boxes", and "objectness_logits",
#                 "gt_classes", "gt_boxes".

#         Returns:
#             In training, a dict of losses.
#             In inference, a list of `Instances`, the predicted instances.
#         """
#         features = [features[f] for f in self.box_in_features]
#         box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
#         box_features = self.box_head(box_features)
#         predictions = self.box_predictor(box_features)  # scores, batch*2; proposal_deltas, batch*4, for box
#         del box_features

#         if self.training:
#             losses = self.box_predictor.losses(predictions, proposals)
#             # proposals is modified in-place below, so losses must be computed first.
#             if self.train_on_pred_boxes:
#                 with torch.no_grad():
#                     pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
#                         predictions, proposals
#                     )
#                     for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
#                         proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
#             return losses
#         else:
#             pred_instances, _ = self.box_predictor.inference(predictions, proposals)
#             return pred_instances

#     def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
#         """
#         Forward logic of the mask prediction branch.

#         Args:
#             features (dict[str, Tensor]): mapping from feature map names to tensor.
#                 Same as in :meth:`ROIHeads.forward`.
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the boxes predicted by R-CNN box head.

#         Returns:
#             In training, a dict of losses.
#             In inference, update `instances` with new fields "pred_masks" and return it.
#         """
#         if not self.mask_on:
#             return {} if self.training else instances

#         if self.training:
#             # head is only trained on positive proposals.
#             instances, _ = select_foreground_proposals(instances, self.num_classes)

#         if self.mask_pooler is not None:
#             features = [features[f] for f in self.mask_in_features]
#             boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
#             features = self.mask_pooler(features, boxes)
#         else:
#             features = {f: features[f] for f in self.mask_in_features}
#         return self.mask_head(features, instances)

#     def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
#         """
#         Forward logic of the keypoint prediction branch.

#         Args:
#             features (dict[str, Tensor]): mapping from feature map names to tensor.
#                 Same as in :meth:`ROIHeads.forward`.
#             instances (list[Instances]): the per-image instances to train/predict keypoints.
#                 In training, they can be the proposals.
#                 In inference, they can be the boxes predicted by R-CNN box head.

#         Returns:
#             In training, a dict of losses.
#             In inference, update `instances` with new fields "pred_keypoints" and return it.
#         """
#         if not self.keypoint_on:
#             return {} if self.training else instances

#         if self.training:
#             # head is only trained on positive proposals with >=1 visible keypoints.
#             instances, _ = select_foreground_proposals(instances, self.num_classes)
#             instances = select_proposals_with_visible_keypoints(instances)

#         if self.keypoint_pooler is not None:
#             features = [features[f] for f in self.keypoint_in_features]
#             boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
#             features = self.keypoint_pooler(features, boxes)
#         else:
#             features = {f: features[f] for f in self.keypoint_in_features}
#         return self.keypoint_head(features, instances)

class AdverOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        num_real_classes: int,  # for adversarial discriminator
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # object or not
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        # adversarial discriminator
        self.adv_cls_score = nn.Linear(input_size, num_real_classes + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.adv_cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.adv_cls_score]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_real_classes"      : cfg.MODEL.ROI_HEADS.NUM_REAL_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x, train_adver=False):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        if train_adver:
            with torch.no_grad():
                scores = self.cls_score(x)
                proposal_deltas = self.bbox_pred(x)
            adv_scores = self.adv_cls_score(x)
            
        else:
            scores = self.cls_score(x)
            proposal_deltas = self.bbox_pred(x)
            with torch.no_grad():
                adv_scores = self.adv_cls_score(x)
        return scores, adv_scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, adv_scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        gt_real_classes = (
            cat([p.gt_real_classes_id for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(adv_scores, gt_real_classes, prefix="fast_rcnn_adv_discrim")

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
            "loss_adv_real_cls": -cross_entropy(adv_scores, gt_real_classes, reduction="mean"),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)       # 按照每张图的instance数，分割preds为每张图一块

