import pytorch_lightning as pl
from src.finetune.detector import multi_stage_models as models
from src.finetune.detector.pseudolabeler import (
    ConsensusLabeler,
    SemanticMapConsensusLabeler,
    SoftConsensusLabeler,
    VanillaConsensusLabeler
)
from typing import List

class TeacherStudent(pl.LightningModule):
    def __init__(
        self,
        detectron_args,
        
        # student_model=None,
        
        freeze_teacher=True,
        use_teacher=False,
        batch_size=1,
        # student model params
        student_thr=0.5,
        # teacher model params
        consensus="vanilla",
        temperature=1,
        thr=0.7,
        solution="ours",
        *args,
        **kwargs,
    ):
        super().__init__()
        
        # TODO: mixup
        self.student_model_cls = models.FocalSoftMultiStageModel
        self.student_thr = student_thr
        self.max_steps = None   # TODO
        # student training params
        self.batch_size = batch_size  #TODO
        
        # teacher model: pseudo labeler
        switch = {
            "logits": SoftConsensusLabeler,
            "vanilla": VanillaConsensusLabeler,
            "semantic_map": SemanticMapConsensusLabeler,
        }
        self.teacher_model: ConsensusLabeler = switch[consensus](
            model=models.MultiStageModel(detectron_args, prune=True),
            temperature=temperature,
            thr=thr,
            solution=solution,
            # device=self.device_id
        )
        
        self.detectron_args = detectron_args
        # self.reinit_online()
        self.save_hyperparameters()
    
    def init_student(self):
        self.student_model = self.student_model_cls(self.detectron_args)
        self.student_model.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            self.student_thr
        )
        
    def training_step(self, batched_inputs, batch_idx):
        for i in batched_inputs:
            if isinstance(i, List):
                batch += i
            else:
                batch.append(i)

        # log
        if batch_idx % 50 == 0:
            self.log_batch(batch, batch_idx)
            
        # train
        losses, _ = self.student_model.training_step(batch, batch_idx)
        
        # log loss:
        for k in losses.keys():
            self.log(
                f"train_{k}",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            
        loss = sum(losses.values())
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss
        
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer = self.student_model.configure_optimizers(max_steps=self.max_steps)   # TODO

        return optimizer
    
    def log_batch(self, batch, batch_idx):
        for idx, x in enumerate(batch):

            remap = BBSense.REMAP
            metadata = MetadataCatalog.get('coco_2017_val')
            visualizer = Visualizer(
                deepcopy(x['image'].permute(1, 2, 0).cpu()),
                metadata,
                instance_mode=ColorMode.IMAGE,
            )
            y = deepcopy(x['instances'])

            y.pred_classes = torch.tensor([remap[p.item()] for p in y.gt_classes])
            if hasattr(y, "gt_boxes"):
                y.pred_boxes = y.gt_boxes
            if hasattr(y, "gt_masks"):
                if isinstance(y.gt_masks, Tensor):
                    y.pred_masks = y.gt_masks
                else:
                    y.pred_masks = y.gt_masks.tensor
            frame = visualizer.draw_instance_predictions(
                predictions=y.to('cpu')
            ).get_image()
            img = wandb.Image(frame)

            self.trainer.logger[0].log_metrics(
                {
                    f"gt-batch-{batch_idx}-img-{idx}": img,
                    "trainer/global_step": self.trainer.global_step,
                }
            )
    
    
    