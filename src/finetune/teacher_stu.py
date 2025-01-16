import pytorch_lightning as pl
from finetune.detector import multi_stage_models as models

class TeacherStudent(pl.LightningModule):
    def __init__(
        self,
        detectron_args,
        consensus="vanilla",
        # student_model=None,
        thr=0.7,
        freeze_teacher=True,
        use_teacher=False,
        batch_size=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        # TODO: labeler
        
        self.student_model_class = models.FocalSoftMultiStageModel
        
        # self.target_network: ConsensusLabeler = switch[consensus](
        #     model=models.MultiStageModel(detectron_args, prune=True),
        #     temperature=temperature,
        #     thr=thr,
        #     solution=solution,
        #     device=self.device_id
        # )
        
        # self.reinit_online()
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        pass
    
    
    
    