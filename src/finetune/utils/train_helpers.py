from itertools import chain
import src
from src.finetune.utils.ddp_long_timeout import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from hydra.utils import get_original_cwd, to_absolute_path
import os

def list_helper_collate(batch):
    return list(chain(*[[elem for elem in elems_list] for elems_list in batch]))

def dict_helper_collate(batch):
    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]

def _get_wandb_logger(exp_name: str, project_name: str):
    logger = WandbLogger(
        name=exp_name,
        project=project_name,
        entity='panpanono'
    )
    return logger

def get_training_params(cfg):
    """

    Parameters

    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    -------

    """

    # log
    logger = [
        _get_wandb_logger(
            project_name=src.project_name,
            exp_name=cfg.exp_name + "/" + cfg.training.exp_name_training,
        ),
    ]
    # exp_path = os.getcwd()          # TODO
    exp_path = os.getcwd()
    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save model
    ckpt_cb = ModelCheckpoint(      
        monitor="val_map_50_online",
        mode='max',
        save_last=True,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        every_n_epochs=2,
    )

    # device
    gpus = cfg["gpus"]

    # other


    trainer_configuration = {
        "multiple_trainloader_mode": "min_size",
        "default_root_dir": checkpoint_dir,
        "gpus": gpus,
        "max_epochs": cfg["training"]["epochs"],
        "callbacks": [ckpt_cb],
        "enable_checkpointing": True,
        "weights_summary": "top",
        "logger": logger,
        "plugins": None,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 2,

    }
    trainer_configuration['strategy'] = DDPPlugin(find_unused_parameters=True)

    # if "debug" in cfg and cfg['debug']:
    #     torch.autograd.set_detect_anomaly(True)
    #     trainer_configuration["overfit_batches"] = 50
    #     trainer_configuration["log_gpu_memory"] = True

    
    if cfg['training']['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="train_loss_cls_epoch",
            min_delta=0.001,
            patience=cfg['training']["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)
    
    if cfg['training']['ema']:
        ema_callback = BYOLMAWeightUpdate(cfg['training'].teacher_momentum)
        trainer_configuration["callbacks"].append(ema_callback)
        
    return trainer_configuration