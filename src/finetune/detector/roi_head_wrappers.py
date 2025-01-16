from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.layers import cross_entropy
from kornia.losses.focal import focal_loss

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

def soft_cross_entropy(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]

loss_switch = {
    'cross_entropy': cross_entropy,
    'focal': focal_loss,
    
}
class MinimalPredictorWrapper(nn.Module):
    '''
        剪切roi_head权重
    '''
    def __init__(self, prediction_head):
        super().__init__()
        assert isinstance(
            prediction_head, (FastRCNNOutputLayers)
        ), "Trying to wrap a ROIHead different from FastRCNNOutputLayer"
        self.box_predictor = prediction_head
    
    def reinit_head(self, classes_idxs):
        classes_to_keep = np.array([*classes_idxs, 80])

        self.box_predictor.cls_score.bias = torch.nn.Parameter(
            self.box_predictor.cls_score.bias[classes_to_keep]
        )
        self.box_predictor.cls_score.weight = torch.nn.Parameter(
            self.box_predictor.cls_score.weight[classes_to_keep]
        )
        classes_to_keep = np.array([*classes_idxs])
        mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
            np.arange(0, 4), len(classes_to_keep)
        )
        self.box_predictor.bbox_pred.weight = torch.nn.Parameter(
            self.box_predictor.bbox_pred.weight[mask]
        )
        self.box_predictor.bbox_pred.bias = torch.nn.Parameter(
            self.box_predictor.bbox_pred.bias[mask]
        )
        self.box_predictor.num_classes = len(classes_idxs)
        self.box_predictor.cls_score.out_features = len(classes_idxs)
        self.box_predictor.bbox_pred.out_features = len(classes_idxs) * 4


    def forward(self, x):
        return self.box_predictor(x)
    
    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)
    

class BoxPredictorWrapper(MinimalPredictorWrapper):
    '''
        指定cls_loss, 默认cross_entropy
    '''
    def __init__(
        self, 
        prediction_head: FastRCNNOutputLayers, 
        cls_loss="cross_entropy", 
        *args, 
        **kwargs
    ):
        super().__init__(prediction_head)
        
        print(f"use {cls_loss} loss")
        loss_func = loss_switch[cls_loss]
        if cls_loss == "focal":
            assert 'focal_loss_alpha' in kwargs and 'focal_loss_gamma' in kwargs, "focal loss need alpha and gamma"
            self.focal_loss_alpha = kwargs['focal_loss_alpha']
            self.focal_loss_gamma = kwargs['focal_loss_gamma']
            self.cls_loss = lambda x, y, reduction: 10* loss_func(  # noqa: E731
                x, y, reduce=False, reduction=reduction
            )
        else:
            self.cls_loss = lambda x, y, reduction: loss_func(  
                x, y, reduction=reduction
            )
        
    def losses(self, predictions, proposals, reduction='mean'):
        """
            重写losses，主要是使用指定cls_loss
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        logits, proposal_deltas = predictions

        gt_classes = (
            torch.cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = torch.cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = torch.cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )

        if len(proposals) == 0:
            cls_loss_ = logits.sum() * 0.0  # connect the gradient

        else:
            cls_loss_ = self.cls_loss(logits, gt_classes, "mean")

        losses = {
            "loss_cls": cls_loss_,
            "loss_box_reg": self.box_predictor.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        return {
            k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()
        }
    
    def inference(self, predictions, proposals):
        pass


class SoftHeadWrapper(BoxPredictorWrapper):
    def __init__(
        self,
        prediction_head,
        cls_loss="cross_entropy",
        temperature=1,
        alpha=0.5,
        # soft_loss=None,
        *args,
        **kwargs
    ):
        super().__init__(prediction_head=prediction_head, cls_loss=cls_loss, *args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        
        self.soft_loss = lambda x, y: soft_cross_entropy(x, y)
        print(f"use distill loss, temperature is {self.temperature}")
    
    def losses(self, predictions, proposals):

        losses = super().losses(predictions, proposals)
        if len(proposals) == 0 or any([hasattr(p, 'gt_logits') for p in proposals]):
            cons_loss = self._get_consistency_loss(predictions, proposals)
            losses['loss_soft'] = cons_loss
        return losses
    
    def _get_consistency_loss(self, predictions, proposals, reduction="mean"):
        # TODO
        logits, proposal_deltas = predictions
        gt_logits = []

        if len(proposals):
            logits_mask = []

            for p in proposals:
                if hasattr(p, 'gt_logits'):
                    gt_logits.append(p[p.gt_classes != len(BBSense.CLASSES)].gt_logits)
                    logits_mask.append(p.gt_classes != len(BBSense.CLASSES))
                else:
                    logits_mask.append(torch.zeros_like(p.gt_classes, dtype=torch.bool))
            logits_mask = torch.cat(logits_mask)
            gt_logits = torch.cat(gt_logits, dim=0)

        else:
            logits_mask = torch.empty()

        if gt_logits.numel() == 0 and reduction == "mean":
            distillation_loss = logits.sum() * 0.0  # connect the gradient

        else:
            # Calculate distillation loss

            soft_log_probs = F.softmax(logits[logits_mask] / self.temperature, dim=-1)      # 模型预测的logits
            # gt_logits: 从pseudo labeler得到
            distillation_loss = self.soft_loss(
                soft_log_probs, gt_logits       
            )  # Note: gt_logits have been already normalized by temperature; 在pseudo label时除以temperature, 赋值给gt_logits
        return distillation_loss * self.alpha


if __name__ == "__main__":
    boxWap = MinimalPredictorWrapper()