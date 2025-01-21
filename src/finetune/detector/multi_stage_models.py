from src.finetune.detector.predictor_utils import Predictor
from src.finetune.detector.roi_head_wrappers import BoxPredictorWrapper, SoftHeadWrapper
from ..sensors_data import BBSense
import torch
from ..utils import triplet
from typing import Dict, List, Optional, Tuple


class MultiStageModel(Predictor):
    def __init__(
        self,
        cfg=None,
        lr=0.0001,
        loss_weights={},
        use_gt_matching=True,
        optimizer="SGD",
        optimizer_params={},
        prune=True,
        compute_loss=True,
        loss_margin=0.3,
        head_cls=BoxPredictorWrapper,
        mask_on=True,
        load_checkpoint=True,
        *args,
        **kwargs,
    ):
        
        # TODO: box reg loss: 'giou'
        super().__init__(cfg, load_checkpoint=load_checkpoint)
        
        assert prune == True, "must prune detection model"
        if prune:  # Prune classifier instead of initializing a new one
            self.reinit_head(BBSense.CLASSES)
        self.set_head_wrapper(head_cls)
        
        self.lr = lr
        self.optimizer = optimizer
        self.opt_params = optimizer_params  # TODO Params for optimizer
        self.loss_weights = loss_weights    # TODO

        self.feature_projector = triplet.tinyprojection_MLP(1024, out_dim=128)

        # Stages losses
        self.compute_head_loss = True
        self.compute_projector_loss = True
        self.compute_proposal_loss = True
        
        self.save_hyperparameters()
        
    def configure_optimizers(self, *args, **kwargs):
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )

        return optimizer
    
    def training_step(self, batch, batch_idx):
        losses, predictions = self._common_step(batch)
        return losses, predictions
    
    def validation_step(self, batch, batch_idx):
        # always use gt for validation
        losses, predictions = self._common_step(batch,)  
        return losses, predictions
    
    def _common_step(self, batch):
        (
            predictions,
            pred_loss,
            box_features,
            y_matching,
        ) = self._compute(batch)
        
        # contrastive loss
        contrastive_loss = None
        if self.loss_weights.get('contrastive_loss', 1.0) > 0:
            # 在所有输入中，所有instances找到最难正样本和负样本，构建triple，计算loss
            contrastive_loss = self._compute_contrastive_loss(box_features, y_matching)
            
            
        result = {}
        if contrastive_loss is not None:
            contrastive_loss = contrastive_loss * self.loss_weights.get(
                'contrastive_loss', 1.0
            )
            result['loss_contrastive'] = contrastive_loss
            
        if pred_loss is not None:

            for key, _ in pred_loss.items():
                pred_loss[key] *= self.loss_weights.get(key, 1.0)
            result = {**result, **pred_loss}
            
        return result, predictions
    
    def _compute_contrastive_loss(self, features, y):
        if self.compute_projector_loss:

            y_mask = y != -1
            y = y[y_mask]

            if len(y) > 1:
                
                features = self.feature_projector(features[y_mask])
                return triplet.online_mine_hard(
                    y, features, self.loss_margin, device=self.device
                )[0]
            else:
                return features.sum() * 0.0  # connect the gradient
        else:
            return None
    
    @torch.no_grad()
    def __call__(self, inputs):
        '''
        called by predict_step() of pseudolabelers
        '''
        self.eval()
        height = inputs[0]['height']
        width = inputs[0]['width']

        images = self.preprocess_image(inputs)

        if "instances" in inputs[0]:
            gt_instances = [x["instances"] for x in inputs]
        else:
            gt_instances = None

        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, gt_instances)

        instances, _ = self.model.roi_heads(images, features, proposals, gt_instances)
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        predictions_images = []

        for i in range(len(instances)):
            predictions_images += [i] * len(instances[i])

        if gt_instances is not None:
            boxes = [gt_instances[i].gt_boxes for i in range(len(gt_instances))]
        else:
            boxes = [instances[i].pred_boxes for i in range(len(instances))]

        pooled_features = self.model.roi_heads.box_pooler(mask_features, boxes)
        box_features = self.feature_projector(
            self.model.roi_heads.box_head(pooled_features)
        )

        predictions = self.postprocess(height, width, instances)

        return predictions, box_features  # , predictions_images
    
    
    def _compute(self, batched_inputs):
        '''
        network forward propagation
        
        '''
        inputs = []
        for i in batched_inputs:
            if isinstance(i, List):
                inputs += i
            else:
                inputs.append(i)

        # get consistent object id
        gt_instances = []
        for x in inputs:
            x = x["instances"]
            if hasattr(x, "infos"):
                x.gt_ids = torch.tensor(
                    [i['id_object'] for i in x.infos], dtype=torch.int16)
            else:
                x.gt_ids = torch.ones(len(x)) * -1
            gt_instances.append(x)
        
        # preprocess
        images = self.preprocess_image(inputs)
        
        ## forward
        features = self.model.backbone(images.tensor)
        
        proposals, prop_loss = self.model.proposal_generator(
            images, features, gt_instances
        )

        # extract features
        mask_features = [features[f] for f in self.model.roi_heads.in_features]

        # 将gt 匹配 proposals, 匹配到的打标
        labeled_props = self.model.roi_heads.label_and_sample_proposals(    
            proposals, gt_instances
        )
        prop_boxes = [p.proposal_boxes for p in labeled_props]
        prop_ids = torch.cat(
            [
                p.gt_ids.to("cpu") if hasattr(p, "gt_ids") else torch.ones(len(p)) * -1
                for p in labeled_props
            ]
        )
        # Get features for the regions proposals
        prop_classes = torch.cat([p.gt_classes for p in labeled_props])

        box_features = self.model.roi_heads.box_pooler(mask_features, prop_boxes)
        box_features = self.model.roi_heads.box_head(box_features)
        y_mask = prop_classes != len(BBSense.CLASSES)  # Background
        box_features = box_features[y_mask]
        if not (prop_ids.device == y_mask.device):
            prop_ids = prop_ids.to(y_mask.device)

        prop_ids = prop_ids[y_mask]

        
        ## head loss
        prediction_loss = {**prop_loss}
        if self.compute_head_loss:
            self.model.roi_heads.train()

            _, head_loss = self.model.roi_heads(
                images, features, proposals, gt_instances
            )
            prediction_loss = {**prediction_loss, **head_loss}

        ## get prediction
        self.model.roi_heads.eval()
        instances = self.head_forward(images, features, proposals)
        self.model.roi_heads.train()
        
        ## postprocess
        height = inputs[0]['height']
        width = inputs[0]['width']
        outputs = self.postprocess(height, width, instances)

        return (
            outputs,
            prediction_loss,
            box_features,
            prop_ids,
        )
    
class SoftMultiStageModel(MultiStageModel):
    '''
        with distill head
    '''
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: SoftHeadWrapper(
                prediction_head=x,
                cls_loss=kwargs.get('cls_loss', 'cross_entropy'),
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.5),
            )
        super().__init__(*args, **kwargs)