from src.finetune.teacher_stu import TeacherStudent
from src.finetune.utils.train_helpers import get_training_params
from torch.utils.data.dataloader import DataLoader
from detectron2.utils.events import EventStorage


class Pipeline:
    teacher_student_model_cls = TeacherStudent
    
    def __init__(self, cfg):
        
        
        self.teacher_student = Pipeline.teacher_student_model_cls(
            **cfg,
            **cfg.training,
        )
        
        self.cfg = cfg
        
        # for iterations
        self.epochs_per_iteration = cfg.training.epochs
    
    def set_trainer_params(self, cfg):
        self.trainer_config = get_training_params(cfg)
        
    def fit_student_and_update_teacher(
        self, 
        dataloader: DataLoader,  
        trainer = None, 
        checkpoint_path: str = None,
    ):
        
        with EventStorage():
            if checkpoint_path:
                trainer.fit(self.teacher_student, dataloader, ckpt_path=checkpoint_path)
            else:
                trainer.fit(self.teacher_student, dataloader)
                
    def save_teacher_and_update_configs(self):
        """
        Trained object-detector becomes new pseudo-labeler. It's attached to
        the policy and the process can start again with a new iteration

        """
        self.pseudo_labeler.to("cpu")
        self.trainer_config['max_epochs'] += self.epochs_per_iteration
        # TODO
        if self.cfg.training['update_target']: 
            if  not self.cfg.training.ema:
                self.pseudo_labeler.reinit(self.teacher_student.student_model)