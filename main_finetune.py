import pytorch_lightning as pl
from src.policy_rl.arguments import get_args
from src.finetune.datamodule import GTDataModule, HabitatDataModule
from src.finetune.dataset_habitat import BbsgtDataset
from src.finetune.pipelines import Pipeline
from src.finetune.dataset_utils import get_loader
from src.finetune.utils.train_helpers import dict_helper_collate
from src.finetune.dataset_utils import get_coco_item_dict, SampleLoader

from detectron2.utils.events import EventStorage

import albumentations as A
import hydra
import torch
import os
import random
import numpy as np

os.environ["WANDB_MODE"]="offline"
@hydra.main(config_path='./configs_finetune/', config_name='train.yaml')
def main(cfg):
    seed = 1234
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    pipeline = Pipeline(cfg)
    
    trainer = pl.Trainer(**pipeline.trainer_config)
    
    if "checkpoint_path" in cfg and cfg.checkpoint_path is not None:
        ckpt_path = cfg.checkpoint_path
        
        if os.path.exists(ckpt_path):
            print(f"load from {ckpt_path}")
            pipeline.teacher_student = pipeline.teacher_student.load_from_checkpoint(ckpt_path)
                
    for id_iteration in range(cfg.n_iterations):
        # dataset
        dataset_path = cfg.sample_path
        if "use_gt" in cfg.training and cfg.training.use_gt:    # TODO?
            dm = GTDataModule(pipeline.pseudo_labeler, dataset_path, 
                            **cfg, **cfg.training)    # TODO
        else:
            dm = HabitatDataModule(pipeline.pseudo_labeler, dataset_path, 
                                **cfg, **cfg.training)   # # TODO

        # training
    
        pipeline.fit_student_and_update_teacher(dm, trainer)
    
        checkpoint_path = f"iteration-{id_iteration}.ckpt"
        trainer.save_checkpoint(checkpoint_path)        # TODO 绝对路径还是i相对路径
        pipeline.save_teacher_and_update_configs()
        
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
    sampler = SampleLoader(cfg.testset_path)
    inputs = sampler.get_env_episode_and_steps_dense_list()     
    filter_empty_instances = []
    for env, ep, step in zip(inputs[0], inputs[1], inputs[2]):      # need long time
        instances = sampler.get_sample(env, ep, step, "bbsgt").get_bbs_as_gt()

        filter_empty_instances.append(len(instances) > 0)       # 仅保留有mask的
    dataset = BbsgtDataset(
            data_path=None,
            sampler=sampler,
            index_mask=filter_empty_instances,      # 仅保留有mask的
            transform=transform,
        )
    test_loader = get_loader(
        dataset,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=10,
        collate_fn=dict_helper_collate,
    )

    
    with EventStorage():
        with torch.no_grad():
            trainer.test(pipeline.teacher_student, test_loader)
    

if __name__ == "__main__":
    main()