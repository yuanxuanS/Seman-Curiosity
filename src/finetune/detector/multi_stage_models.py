from src.finetune.detector.predictor_utils import Predictor
from src.finetune.detector.roi_head_wrappers import BoxPredictorWrapper, SoftHeadWrapper
from ..sensors_data import BBSense
import torch


class MultiStageModel(Predictor):
    def __init__(
        self,
        cfg=None,
        lr=0.01,
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
                
        self.lr = lr
        self.optimizer = optimizer

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
        pass
    
    
class FocalSoftMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: SoftHeadWrapper(
                prediction_head=x,
                cls_loss="focal",
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.5),
            )
        super().__init__(*args, **kwargs)