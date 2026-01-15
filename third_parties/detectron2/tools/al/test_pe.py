import warnings
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, launch, default_setup
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOALEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch
import logging
import os 
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

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
    # 打印一下路径，确认在日志里能看到它到底传了什么
    print(f"Loading config from: {args.config_file}")
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        if cfg.INFER or cfg.INFER_DIVERSITY:
            evaluator_list.append(COCOALEvaluator(dataset_name, output_dir=output_folder))
        else:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)




def do_test(cfg, model):
    
    # 主动学习的数据集，指定json标签
    from detectron2.data.datasets import register_coco_instances
    if cfg.INFER or cfg.INFER_DIVERSITY:
        print("Inference mode")
        image_root = cfg.IMG_ROOT
        dataset_name = f"proj_active_round_{cfg.ROUND_IDX}"
        logger.info(f"data json is {cfg.DATA_JSON}")
        register_coco_instances(dataset_name, {}, cfg.DATA_JSON, image_root)
        cfg.defrost()
        cfg.DATASETS.TEST = (dataset_name,)
        cfg.freeze()
    else:
        cfg.defrost()
        cfg.DATASETS.TEST = ('proj_test',)
        cfg.freeze()
    
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        prefix = "diversity_" if cfg.INFER_DIVERSITY else "uncertainty_"
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, prefix+dataset_name)
        )
        
        results_i = inference_on_dataset(model, data_loader, evaluator)

        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
        
    #
    return results




def main(args):
    cfg = setup(args)
    keep_class = { 
    
    2: "car",       # key为COCO原数据类别中的id, 顺序对应类别的顺序
    1: "bicylcle",
    13: "bench",
    10: "fire hydrant",
    
    
    # 56: "chair",
    # 57: "couch",
    # 59: "bed",
    # 61: "toilet",
    # 72: "refrigerator",
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
    
    # 直接load不用prune extend， 但是要该类别数
    cfg.defrost()
    cfg.MODEL.RETINANET.NUM_CLASSES = len(keep_class) + len(extend_class)
    cfg.freeze()       
    # 
    
    model = build_model(cfg)
    
    
    logger.info("Model:\n{}".format(model))
    
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )        
    return do_test(cfg, model)



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Automatically injected by torch.distributed.launch")
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
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
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
