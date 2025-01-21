import pytorch_lightning as pl
from arguments import get_args
from pipelines import GTDataModule, HabitatDataModule

def main():
    args = get_args()
    
    pipeline = pipelines.Pipeline(args)
    
    trainer = pl.Trainer(**pipeline.trainer_config)
    
    # dataset
    if args.training == "use_gt":
        dm = GTDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, **args, **args.training)
    else:
        dm = HabitatDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, **args, **cfg.training)

    # training
    pipeline.fit_student_and_update_teacher(dm, trainer)
    id_iteration = 0
    checkpoint_path = f"iteration-{id_iteration}.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    
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
    dataset = SinglecamEpisodeDetectionHabitatObjectsDataset(
            os.path.join(cfg.data_base_dir, "fix_test"),
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