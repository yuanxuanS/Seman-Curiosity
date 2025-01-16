from finetune.teacher_stu import TeacherStudent


class Pipeline:
    teacher_student_model_cls = TeacherStudent
    
    def __init__(self, cfg):
        
        
        self.teacher_student = self.teacher_student_model_class(
            habitat=cfg.habitat_cfg,        # TODO
            **cfg,
            **cfg.training,
        )
        
        self.cfg = cfg
    
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