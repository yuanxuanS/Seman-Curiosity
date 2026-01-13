# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "AdverRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], return_features=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, return_features=return_features)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        return_features: bool = False,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            if return_features:
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), features
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            if return_features:
                return results, features
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"][:, :, :].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def extract_feature(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return features

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class AdverRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(backbone=backbone,
                         proposal_generator=proposal_generator,
                         roi_heads=roi_heads,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         input_format=input_format,
                         vis_period=vis_period)
        self.adver_part = self.roi_heads.box_predictor

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], train_adver=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if train_adver: # if train adversarial discrimintor, frozon main part
            with torch.no_grad():
                features = self.backbone(images.tensor)
                if self.proposal_generator is not None:
                    proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                    proposal_losses = {}
        else:   # if train main part, not frozon it
            features = self.backbone(images.tensor)
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, train_adver)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


@META_ARCH_REGISTRY.register()
class PruneRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(backbone=backbone,
                         proposal_generator=proposal_generator,
                         roi_heads=roi_heads,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         input_format=input_format,
                         vis_period=vis_period)
        # self.adver_part = self.roi_heads.box_predictor
    
    def reinit_head(self, classes_idxs):    # TODO
        self.roi_heads.num_classes = len(classes_idxs)

        # if isinstance(self.model.roi_heads.box_predictor, MinimalPredictorWrapper):
        #     self.model.roi_heads.box_predictor.reinit_head(classes_idxs)
        # else:
        classes_to_keep = np.array([*classes_idxs, 80])
        cls_bias = torch.nn.Parameter(
            self.roi_heads.box_predictor.cls_score.bias[classes_to_keep]
        )
        cls_weight = torch.nn.Parameter(
            self.roi_heads.box_predictor.cls_score.weight[classes_to_keep]
        )

        classes_to_keep = np.array([*classes_idxs])
        mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
            np.arange(0, 4), len(classes_to_keep)
        )

        box_weight = torch.nn.Parameter(
            self.roi_heads.box_predictor.bbox_pred.weight[mask]
        )
        box_bias = torch.nn.Parameter(
            self.roi_heads.box_predictor.bbox_pred.bias[mask]
        )
        self.roi_heads.box_predictor.num_classes = len(classes_idxs)

        in_features = box_weight.shape[1]
        self.roi_heads.box_predictor.cls_score = nn.Linear(
            in_features, len(classes_to_keep) + 1
        )
        self.roi_heads.box_predictor.cls_score.bias = cls_bias
        self.roi_heads.box_predictor.cls_score.weight = cls_weight

        self.roi_heads.box_predictor.bbox_pred = nn.Linear(
            in_features, len(classes_idxs) * 4
        )
        self.roi_heads.box_predictor.bbox_pred.bias = box_bias
        self.roi_heads.box_predictor.bbox_pred.weight = box_weight

        if hasattr(self.roi_heads, "mask_head"):
            classes_to_keep = np.array([*classes_idxs])
            mask_weight = torch.nn.Parameter(
                self.roi_heads.mask_head.predictor.weight[classes_to_keep]
            )
            mask_bias = torch.nn.Parameter(
                self.roi_heads.mask_head.predictor.bias[classes_to_keep]
            )
            self.roi_heads.mask_head.predictor.weight = mask_weight
            self.roi_heads.mask_head.predictor.bias = mask_bias
            self.roi_heads.mask_head.predictor.num_classes = len(classes_idxs)
    
    def extend_head(self, num_new_classes):
        """
        在现有子类基础上增加 m 个新类别
        num_new_classes: 增加的类别数量 m
        """
        # 1. 获取基础信息
        device = self.roi_heads.box_predictor.cls_score.weight.device
        old_num_classes = self.roi_heads.box_predictor.num_classes # 当前已有的 n 个类
        new_total_classes = old_num_classes + num_new_classes
        in_features = self.roi_heads.box_predictor.cls_score.in_features

        # --- 分类层 (cls_score) 权重处理 ---
        # 提取旧权重：前 n 个是物体，最后一个是背景
        old_cls_weight = self.roi_heads.box_predictor.cls_score.weight.data
        old_cls_bias = self.roi_heads.box_predictor.cls_score.bias.data
        
        # 分离物体权重和背景权重
        old_obj_w = old_cls_weight[:old_num_classes]
        old_obj_b = old_cls_bias[:old_num_classes]
        bg_w = old_cls_weight[-1:]
        bg_b = old_cls_bias[-1:]

        # 初始化新类别的权重 (使用随机初始化或零初始化)
        new_obj_w = torch.randn(num_new_classes, in_features, device=device) * 0.01
        new_obj_b = torch.zeros(num_new_classes, device=device)

        # 拼接：[旧物体, 新物体, 背景]
        final_cls_w = torch.cat([old_obj_w, new_obj_w, bg_w], dim=0)
        final_cls_b = torch.cat([old_obj_b, new_obj_b, bg_b], dim=0)

        # 重新创建分类层
        self.roi_heads.box_predictor.cls_score = nn.Linear(
            in_features, new_total_classes + 1
        ).to(device)
        self.roi_heads.box_predictor.cls_score.weight = nn.Parameter(final_cls_w)
        self.roi_heads.box_predictor.cls_score.bias = nn.Parameter(final_cls_b)

        # --- 回归层 (bbox_pred) 权重处理 ---
        # 检查是否为 Class-Specific (输出维度 > 4)
        if self.roi_heads.box_predictor.bbox_pred.out_features > 4:
            old_box_w = self.roi_heads.box_predictor.bbox_pred.weight.data
            old_box_b = self.roi_heads.box_predictor.bbox_pred.bias.data

            # 初始化新类别的回归权重
            new_box_w = torch.randn(num_new_classes * 4, in_features, device=device) * 0.001
            new_box_b = torch.zeros(num_new_classes * 4, device=device)

            final_box_w = torch.cat([old_box_w, new_box_w], dim=0)
            final_box_b = torch.cat([old_box_b, new_box_b], dim=0)

            self.roi_heads.box_predictor.bbox_pred = nn.Linear(
                in_features, new_total_classes * 4
            ).to(device)
            self.roi_heads.box_predictor.bbox_pred.weight = nn.Parameter(final_box_w)
            self.roi_heads.box_predictor.bbox_pred.bias = nn.Parameter(final_box_b)

        # --- Mask Head 处理 ---
        if hasattr(self.roi_heads, "mask_head"):
            mask_predictor = self.roi_heads.mask_head.predictor
            old_mask_w = mask_predictor.weight.data
            old_mask_b = mask_predictor.bias.data
            
            # Mask Head 只有物体通道，直接在后面拼
            new_mask_w = torch.randn(
                num_new_classes, mask_predictor.in_channels, 
                *mask_predictor.kernel_size, device=device
            ) * 0.01
            new_mask_b = torch.zeros(num_new_classes, device=device)

            final_mask_w = torch.cat([old_mask_w, new_mask_w], dim=0)
            final_mask_b = torch.cat([old_mask_b, new_mask_b], dim=0)

            # 假设是 Conv2d，重新初始化
            new_m_predictor = nn.Conv2d(
                mask_predictor.in_channels, new_total_classes,
                mask_predictor.kernel_size, mask_predictor.stride, mask_predictor.padding
            ).to(device)
            new_m_predictor.weight = nn.Parameter(final_mask_w)
            new_m_predictor.bias = nn.Parameter(final_mask_b)
            
            self.roi_heads.mask_head.predictor = new_m_predictor
            self.roi_heads.mask_head.num_classes = new_total_classes

        # 更新元数据
        self.roi_heads.num_classes = new_total_classes
        self.roi_heads.box_predictor.num_classes = new_total_classes

    def reinit_head_list(self, classes_idxs):
        # 1. 确定基本参数
        # 假设原始模型是 COCO 80类，则背景索引为 80
        old_bg_idx = 80 
        new_num_classes = len(classes_idxs)
        device = next(self.parameters()).device
        
        # 2. 遍历 Cascade R-CNN 的所有预测阶段
        # Cascade R-CNN 的 predictors 存储在 self.roi_heads.box_predictor (是一个 ModuleList)
        for i, predictor in enumerate(self.roi_heads.box_predictor):
            
            # --- 处理分类层 (cls_score) ---
            in_features = predictor.cls_score.in_features
            
            # 提取指定类和背景类的权重
            # 索引选择：[class0, class1, ..., old_background]
            keep_idxs = torch.tensor([*classes_idxs, old_bg_idx], device=device)
            
            with torch.no_grad():
                new_cls_weight = predictor.cls_score.weight[keep_idxs].clone()
                new_cls_bias = predictor.cls_score.bias[keep_idxs].clone()
                
                # 重新初始化该阶段的 Linear 层
                # 输出维度 = 新类别数 + 1 (背景)
                predictor.cls_score = nn.Linear(in_features, new_num_classes + 1).to(device)
                predictor.cls_score.weight.copy_(new_cls_weight)
                predictor.cls_score.bias.copy_(new_cls_bias)
            
            # --- 处理回归层 (bbox_pred) ---
            # 重点：根据你的代码断言，Cascade R-CNN 是 CLS_AGNOSTIC
            # 如果是 Class-Agnostic，out_features 应该是 4，不需要任何剪切
            # 如果你强制修改了配置为 Class-Specific，才需要执行类似 Faster R-CNN 的 mask 逻辑
            if predictor.bbox_pred.out_features > 4:
                # 只有在非 Class-Agnostic 模式下才运行这里的逻辑
                mask = []
                for c in classes_idxs:
                    mask.extend(range(c * 4, (c + 1) * 4))
                mask = torch.tensor(mask, device=device)
                
                with torch.no_grad():
                    new_box_weight = predictor.bbox_pred.weight[mask].clone()
                    new_box_bias = predictor.bbox_pred.bias[mask].clone()
                    
                    predictor.bbox_pred = nn.Linear(in_features, new_num_classes * 4).to(device)
                    predictor.bbox_pred.weight.copy_(new_box_weight)
                    predictor.bbox_pred.bias.copy_(new_box_bias)

            # 更新 predictor 内部的 num_classes 属性
            predictor.num_classes = new_num_classes

        # 3. 更新 ROIHeads 整体的 num_classes
        self.roi_heads.num_classes = new_num_classes
        
        # 4. 如果有 Mask Head，处理方式与 Faster R-CNN 一致
        if hasattr(self.roi_heads, "mask_head"):
            # ... 这里的逻辑保持不变 ...
            print("exists mask head")
            # 获取设备信息
            mask_predictor = self.roi_heads.mask_head.predictor
            device = mask_predictor.weight.device
            
            # Mask Head 的输出通常是 (num_classes, C, H, W)
            # 它通常不包含背景类，所以直接取指定的 classes_idxs
            classes_to_keep = torch.tensor([*classes_idxs], device=device)
            
            with torch.no_grad():
                # 提取权重和偏置
                # weight shape: [num_classes, C, K, K]
                # bias shape: [num_classes]
                new_mask_weight = mask_predictor.weight[classes_to_keep].clone()
                new_mask_bias = mask_predictor.bias[classes_to_keep].clone()
                
                # 获取输入通道数和其他卷积参数
                in_channels = mask_predictor.in_channels
                out_channels = len(classes_idxs)
                kernel_size = mask_predictor.kernel_size
                stride = mask_predictor.stride
                padding = mask_predictor.padding
                
                # 重新初始化预测层
                # 注意：这里要区分是 Conv2d 还是 ConvTranspose2d (通常是后者进行上采样)
                if isinstance(mask_predictor, nn.ConvTranspose2d):
                    new_predictor = nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size, stride, padding
                    ).to(device)
                else:
                    new_predictor = nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride, padding
                    ).to(device)
                
                # 拷贝权重
                new_predictor.weight.copy_(new_mask_weight)
                new_predictor.bias.copy_(new_mask_bias)
                
                # 替换原有的 predictor
                self.roi_heads.mask_head.predictor = new_predictor
                
                # 更新 mask_head 中的类别属性
                self.roi_heads.mask_head.num_classes = len(classes_idxs)
                
    def extend_head_list(self, num_new_classes, init_std=0.01):
        """
        用于head是list的，比如cascade rcnn
        Args:
            num_new_classes (int): 需要增加的类别数量 m
            init_std (float): 新类别的初始化标准差
        """
        device = next(self.parameters()).device
        
        # 遍历所有阶段的 Predictors
        for i, predictor in enumerate(self.roi_heads.box_predictor):
            old_num_classes = predictor.num_classes  # 这里的 n
            new_total_classes = old_num_classes + num_new_classes # n + m
            in_features = predictor.cls_score.in_features
            
            # --- 1. 扩展分类层 (cls_score) ---
            old_cls_w = predictor.cls_score.weight.data # [n+1, in_features]
            old_cls_b = predictor.cls_score.bias.data   # [n+1]
            
            # 创建新的分类层 (n + m + 1)
            new_cls_score = nn.Linear(in_features, new_total_classes + 1).to(device)
            
            # 初始化
            nn.init.normal_(new_cls_score.weight, std=init_std)
            nn.init.constant_(new_cls_score.bias, 0)
            
            with torch.no_grad():
                # 拷贝前 n 个指定类别的权重
                new_cls_score.weight[:old_num_classes].copy_(old_cls_w[:old_num_classes])
                new_cls_score.bias[:old_num_classes].copy_(old_cls_b[:old_num_classes])
                
                # 拷贝原有的背景类权重到最后一个位置
                new_cls_score.weight[-1].copy_(old_cls_w[-1])
                new_cls_score.bias[-1].copy_(old_cls_b[-1])
                
            predictor.cls_score = new_cls_score

            # --- 2. 扩展回归层 (bbox_pred) ---
            if predictor.bbox_pred.out_features > 4:  # Class-Specific
                old_box_w = predictor.bbox_pred.weight.data
                old_box_b = predictor.bbox_pred.bias.data
                
                new_bbox_pred = nn.Linear(in_features, new_total_classes * 4).to(device)
                nn.init.normal_(new_bbox_pred.weight, std=init_std)
                nn.init.constant_(new_bbox_pred.bias, 0)
                
                with torch.no_grad():
                    # 拷贝前 n 个类的回归权重
                    new_bbox_pred.weight[:old_num_classes * 4].copy_(old_box_w)
                    new_bbox_pred.bias[:old_num_classes * 4].copy_(old_box_b)
                
                predictor.bbox_pred = new_bbox_pred
                
            # 更新类别属性
            predictor.num_classes = new_total_classes

        # --- 3. 扩展 Mask Head (如果有) ---
        if hasattr(self.roi_heads, "mask_head"):
            mask_predictor = self.roi_heads.mask_head.predictor
            old_mask_w = mask_predictor.weight.data
            old_mask_b = mask_predictor.bias.data
            
            # 假设是 ConvTranspose2d
            out_channels = new_total_classes
            new_mask_predictor = nn.Conv2d(
                mask_predictor.in_channels, out_channels, 
                mask_predictor.kernel_size, mask_predictor.stride, mask_predictor.padding
            ).to(device)
            
            nn.init.normal_(new_mask_predictor.weight, std=init_std)
            nn.init.constant_(new_mask_predictor.bias, 0)
            
            with torch.no_grad():
                new_mask_predictor.weight[:old_num_classes].copy_(old_mask_w)
                new_mask_predictor.bias[:old_num_classes].copy_(old_mask_b)
                
            self.roi_heads.mask_head.predictor = new_mask_predictor
            self.roi_heads.mask_head.num_classes = new_total_classes

        # 更新整体属性
        self.roi_heads.num_classes = new_total_classes
