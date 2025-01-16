from itertools import chain
import src

def list_helper_collate(batch):
    return list(chain(*[[elem for elem in elems_list] for elems_list in batch]))

def dict_helper_collate(batch):
    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]

def get_training_params(cfg):
    """

    Parameters

    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    -------

    """

    logger = [
        _get_wandb_logger(
            project_name=src.project_name,
            exp_name=cfg.training.exp_base_name + "/" + cfg.exp_name,
        ),
    ]
    exp_path = os.getcwd()
    # checkpoint_dir = os.path.join(exp_path, "checkpoints")
    # log_profiler = os.path.join(exp_path, "profile.txt")
    # os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_map_50_online",
        mode='max',
        save_last=True,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        every_n_epochs=2,
    )

    gpus = cfg["gpus"]

    if "plugins" in cfg:
        plugins = cfg['plugins']
    else:
        plugins = None



    trainer_configuration = {
        "multiple_trainloader_mode": "min_size",
        "default_root_dir": checkpoint_dir,
        "gpus": gpus,
        "max_epochs": cfg["epochs"] if "epochs" in cfg else 100,
        "callbacks": [ckpt_cb],
        "enable_checkpointing": True,
        "weights_summary": "top",
        "logger": logger,
        "plugins": plugins,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 2,

    }
    trainer_configuration['strategy'] = DDPPlugin(find_unused_parameters=True)

    if "debug" in cfg and cfg['debug']:
        torch.autograd.set_detect_anomaly(True)
        trainer_configuration["overfit_batches"] = 50
        trainer_configuration["log_gpu_memory"] = True

    if "early_stopping" in cfg and cfg['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="train_loss_cls_epoch",
            min_delta=0.001,
            patience=cfg["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)

    return trainer_configuration