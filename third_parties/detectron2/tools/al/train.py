# 用于Active leanring的每轮的train
# # Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy
import argparse
import copy
import os
import os.path as osp
import sys
import time
import warnings

import mmcv


from detectron2.engine import (default_setup, 
                                        default_writers,
                                        default_argument_parser,
                                        launch)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.events import EventStorage
from detectron2.config import get_cfg
from torch.nn.parallel import DistributedDataParallel
import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger("detectron2")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # 新增key
    cfg.LABELED_DATA = ""
    cfg.UNLABELED_DATA = ""
    cfg.ROUND_IDX = 0
    cfg.DATA_JSON = ""
    cfg.IMG_ROOT = ""
    cfg.INFER = False
    
    cfg.MODEL.RETINANET.BASE_MOMENTUM = 0.
    cfg.MODEL.RETINANET.QUALITY_XI = 0.
    
    # diversity
    cfg.INFER_DIVERSITY = False
    cfg.MODEL.HEAD = ""
    cfg.MAX_DET = 0
    cfg.FEAT_DIM = 0
    cfg.TOTAL_IMAGES = 0
    cfg.OUTPUT_PATH = ""
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # 注册新数据集
    from detectron2.data.datasets import register_coco_instances
    image_root = cfg.IMG_ROOT
    dataset_name = f"coco_active_round_{cfg.ROUND_IDX}"
    register_coco_instances(dataset_name, {}, cfg.DATA_JSON, image_root)
    cfg.defrost()
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.freeze()
    
    data_loader = build_detection_train_loader(cfg)
    print(f"Dataset size: {len(data_loader.dataset.dataset)}")
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            


def main(args):
    cfg = setup(args)
    
    
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    model.train()
    
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    
    do_train(cfg, model, resume=args.resume)
    

            
if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Automatically injected by torch.distributed.launch")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )