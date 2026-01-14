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
    
    cfg.DATA_TYPE= ""
    
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
        -1 + 1
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
    
    ## 剪切模型权重
    cfg.defrost()
    cfg.MODEL.RETINANET.NUM_CLASSES = 80
    cfg.freeze()
    model_tmp = build_model(cfg)
    
    keep_class = { 
    
    2: "car",       # key为COCO原数据类别中的id, 从0开始顺序数；
    1: "bicylcle",
    13: "bench",
    10: "fire hydrant",
}
    extend_class = {    #把旧的n个类放在最前面，中间插入m 个新类，最后把旧的背景权重挪到第 n+m的位置。
    1: "building",
    2: "grass",
    3: "fence",
    4: "billboard",
    5: "street light",
    6: "tree",
    7: "basketball stands",
    8: "dustbin",
    9: "statue"
    }
    # 
    # 主动学习需要加载权重
    logger.info("加载已有权重".format(model_tmp))
    DetectionCheckpointer(model_tmp, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
    
    if args.prune:
        if cfg.MODEL.ROI_HEADS.NAME == "CascadeROIHeads":
            model_tmp.reinit_head_list(keep_class)
        else:
            model_tmp.reinit_head(keep_class)
        
        cfg.defrost()
        cfg.MODEL.RETINANET.NUM_CLASSES = len(keep_class)
        cfg.freeze()
    
    if args.extend_cls:
        if cfg.MODEL.ROI_HEADS.NAME == "CascadeROIHeads":
            model_tmp.extend_head_list(len(extend_class))
        else:
            model_tmp.extend_head(len(extend_class))

        cfg.defrost()
        cfg.MODEL.RETINANET.NUM_CLASSES = len(keep_class) + len(extend_class)
        cfg.freeze()
    
    cfg.defrost()
    cfg.MODEL.RETINANET.NUM_CLASSES = 13
    cfg.freeze()
    model = build_model(cfg)
    
    tmp_state_dict = model_tmp.state_dict()
    # 2. 定义需要过滤的关键字
    keys_to_ignore = ["class_momentum", "class_quality"]
    for key in list(tmp_state_dict.keys()):
        if any(ignore_key in key for ignore_key in keys_to_ignore):
            logger.info(f"Removing {key} from state_dict due to size mismatch.")
            tmp_state_dict.pop(key)
    model.load_state_dict(tmp_state_dict, strict=False)
    
    logger.info("赋值剪切后权重".format(model))
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
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Whether to prune the model.",
    )
    parser.add_argument(
        "--extend-cls",
        action="store_true",
        help="Whether to extend the model.",
    )
    
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