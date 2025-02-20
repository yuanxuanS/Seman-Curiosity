import pytorch_lightning as pl
from src.finetune.detector import multi_stage_models as models
from src.finetune.detector.pseudolabeler import (
    ConsensusLabeler,
    SemanticConsensusLabeler,
    LogitsConsensusLabeler,
    VanillaConsensusLabeler
)
from .sensors_data import BBSense
from torchmetrics.detection.map import MAP

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from copy import deepcopy
from typing import List
import torch
from torch import Tensor
import wandb

class TeacherStudent(pl.LightningModule):
    def __init__(
        self,
    # student model params
        detectron_args,
        batch_size=1,
        student_test_thr=0.5,
    # teacher model params
        freeze_teacher=True,
        use_teacher=False,
        consensus="vanilla",
        temperature=1,
        teacher_pred_thr=0.7,
        solution="ours",
        *args,
        **kwargs,
    ):
        super().__init__()
        
        # TODO: mixup
        self.student_model_cls = models.SoftMultiStageModel
        self.student_test_thr = student_test_thr
        self.max_steps = None   # TODO
        # student training params
        self.batch_size = batch_size  #TODO
        
        # teacher model: pseudo labeler
        switch = {
            "logits": LogitsConsensusLabeler,
            "vanilla": VanillaConsensusLabeler,
            "semantic": SemanticConsensusLabeler,
        }
        self.teacher_model: ConsensusLabeler = switch[consensus](
            model=models.MultiStageModel(detectron_args, prune=True, **kwargs),
            solution=solution,
            temperature=temperature,
            thr=teacher_pred_thr,
            
        )
        self.use_teacher = use_teacher
        
        self.detectron_args = detectron_args
        
        self.online_val_map_metric = MAP(class_metrics=True)
        self.test_map_metric = MAP(class_metrics=True)
        
        self.teacher_val_map_metric = MAP(class_metrics=True)
        
        self.kwargs = kwargs
        
        self.init_student()
        
        self.save_hyperparameters()
        
    def init_student(self):
        self.student_model = self.student_model_cls(self.detectron_args, **self.kwargs)
        self.student_model.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            self.student_test_thr
        )
        
    def training_step(self, batched_inputs, batch_idx):
        batch = []
        for i in batched_inputs:
            if isinstance(i, List):
                batch += i
            else:
                batch.append(i)

        # log
        if batch_idx % 50 == 0:
            self.log_batch(batch, batch_idx)
                    
            
        # train
        self.student_model.train()
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
        if self.use_teacher:
            self.tch_validation_step(batch, batch_idx)
        self.stu_validation_step(batch, batch_idx)
        
    def tch_validation_step(self, batch, batch_idx):
        self.teacher_model.eval()
        losses, predictions = self.teacher_model.validation_step(batch, batch_idx)
        
        loss = sum(losses.values())
        for k in losses.keys():

            self.log(
                f"val_{k}_teacher",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            
        self.log(
            'val_loss_teacher',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        
        self._val_map_teacher(batch, predictions)
    
    def _val_map_teacher(self, batch, predictions):

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
            }
            for b in predictions
        ]
        self.teacher_val_map_metric.update(pred, gt)
        
    def stu_validation_step(self, batch, batch_idx):
        self.student_model.eval()
        losses, predictions = self.student_model.validation_step(batch, batch_idx)
        
        loss = sum(losses.values())
        for k in losses.keys():
            self.log(
                f"val_{k}_online",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            'val_loss_online',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self._val_map(batch, predictions)
        
    def _val_map(self, batch, predictions):

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
            }
            for b in predictions
        ]
        self.online_val_map_metric.update(pred, gt)
        
    def validation_epoch_end(self, outputs):
        # self.online_val_map_metric = self.online_val_map_metric.to(self.device_id)
        results = self.online_val_map_metric.compute()
        for k in results.keys():
            self.log(
                f"val_{k}_epoch",
                results[k],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)
        
        if self.use_teacher:        # TODO, debug
            results_tch = self.teacher_val_map_metric.compute()
            for k in results_tch.keys():
                self.log(
                    f"val_teacher_{k}_epoch",
                    results_tch[k],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
            self.teacher_val_map_metric = MAP(class_metrics=True)
            self.teacher_val_map_metric.to(self.device)
    
    def test_step(self, batch, batch_idx):
        self.student_model.eval()
        _, predictions = self.student_model.validation_step(batch, batch_idx)

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
            }
            for b in predictions
        ]

        self.test_map_metric.update(pred, gt)
        
    def on_test_epoch_end(self):
        results = self.test_map_metric.compute()
        for k in results.keys():
            self.log(
                f"test_{k}",
                results[k],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
    
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
    
    
    