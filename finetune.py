import pytorch_lightning as pl
from src.policy_rl.arguments import get_args
from src.finetune.datamodule import GTDataModule, HabitatDataModule
from src.finetune.dataset import BbsgtDataset
from src.finetune.pipelines import Pipeline
from src.finetune.dataset_utils import get_loader
from src.finetune.utils.train_helpers import dict_helper_collate

from detectron2.utils.events import EventStorage

import albumentations as A
import hydra
import torch
import os

@hydra.main(config_path='./configs_finetune/', config_name='train.yaml')
def main(cfg):
    
    pipeline = Pipeline(cfg)
    
    trainer = pl.Trainer(**pipeline.trainer_config)
    
    # dataset
    dataset_path = cfg.sample_path
    if cfg.training == "use_gt":    # TODO?
        dm = GTDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, 
                          **cfg, **cfg.training)    # TODO
    else:
        dm = HabitatDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, 
                               **cfg, **cfg.training)   # # TODO

    # training
    pipeline.fit_student_and_update_teacher(dm, trainer)
    id_iteration = 0
    checkpoint_path = f"iteration-{id_iteration}.ckpt"
    trainer.save_checkpoint(checkpoint_path)        # TODO 绝对路径还是i相对路径
    
    # testing
    transform = A.Compose(
        [
            A.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels', 'infos'],
        ),
    )
    dataset = BbsgtDataset(
            data_path=os.path.join(cfg.test_path),
            transform=transform,
            remap_classes=True,
        )

    test_loader = get_loader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=10,
        collate_fn=dict_helper_collate,
    )

    with EventStorage():
        with torch.no_grad():
            trainer.test(pipeline.teacher_student, test_loader)
    

if __name__ == "__main__":
    main()