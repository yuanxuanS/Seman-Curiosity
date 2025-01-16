from src.finetune.teacher_stu import TeacherStudent
from src.finetune.utils.train_helpers import get_training_params
from torch.utils.data.dataloader import DataLoader
from detectron2.utils.events import EventStorage


class Pipeline:
    teacher_student_model_cls = TeacherStudent
    
    def __init__(self, cfg):
        
        
        self.teacher_student = Pipeline.teacher_student_model_cls(
            habitat=cfg.habitat_cfg,        # TODO
            **cfg,
            **cfg.training,
        )
        
        self.cfg = cfg
    
    def set_trainer_params(self, cfg):
        self.trainer_config = get_training_params(cfg)
        
    def fit_student_and_update_teacher(
        self, dataloader: DataLoader, checkpoint_path: str = None, trainer = None
    ):
        
        with EventStorage():
            if checkpoint_path:
                trainer.fit(self.teacher_student, dataloader, ckpt_path=checkpoint_path)
            else:
                trainer.fit(self.teacher_student, dataloader)
                
    def save_teacher_and_update_configs(self):
        
        pass