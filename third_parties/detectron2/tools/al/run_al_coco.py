import sys
import os
from detectron2.config import CfgNode as CN

# from mmdet.ppal.sampler import  *
from detectron2.ppal.sampler import *
from detectron2.ppal.builder import build_al_sampler
from detectron2.ppal.utils.running_checks import sys_echo
from detectron2.ppal.utils.running_checks import (
    display_latest_results,
    command_with_time,
    sys_echo
)
import warnings
warnings.filterwarnings("ignore")


import argparse
parser = argparse.ArgumentParser(description='Active learning arguments')
parser.add_argument('--config', required=True, type=str,
                    default='./al_configs/coco/ppal_retinanet_coco.yaml',
                    help='active learning config')
parser.add_argument('--resume', required=False, type=bool, default=False, help='whether to resume training')
parser.add_argument('--model', required=True, type=str, help='running model')
args = parser.parse_args()


cfg = CN()
cfg.VERSION = 2

# Paths
cfg.CONFIG_DIR = ''
cfg.WORK_DIR = ''
cfg.IMG_ROOT = ''
cfg.SEED = 0

# Environment setting
cfg.PYTHON_PATH = ''
cfg.PORT = 0
cfg.GPUS = 0

# Config setting
cfg.TRAIN_CONFIG = ''
cfg.UNCERTAINTY_INFER_CONFIG = ''
cfg.DIVERSITY_INFER_CONFIG = ''

# Active learning setting
cfg.ROUND_NUM = 0
cfg.BUDGET = 0
cfg.BUDGET_EXPAND_RATIO = 0
cfg.OUTPUT_DIR = ''

# Data setting
cfg.ORACLE_PATH = ""
cfg.INIT_LABELED_JSON = ""
cfg.INIT_UNLABELED_JSON = ""
cfg.INIT_MODEL = None

#sampler config

cfg.UNCERTAINTY_SAMPLER_CONFIG = CN()
cfg.UNCERTAINTY_SAMPLER_CONFIG.NAME = ''
cfg.UNCERTAINTY_SAMPLER_CONFIG.n_sample_images = 0
cfg.UNCERTAINTY_SAMPLER_CONFIG.oracle_annotation_path = ''
cfg.UNCERTAINTY_SAMPLER_CONFIG.score_thr = 0.
cfg.UNCERTAINTY_SAMPLER_CONFIG.class_weight_ub = 0.
cfg.UNCERTAINTY_SAMPLER_CONFIG.class_weight_alpha = 0.
cfg.UNCERTAINTY_SAMPLER_CONFIG.dataset_type = ''

cfg.DIVERSITY_SAMPLER_CONFIG = CN()
cfg.DIVERSITY_SAMPLER_CONFIG.NAME = ''
cfg.DIVERSITY_SAMPLER_CONFIG.n_sample_images = 0
cfg.DIVERSITY_SAMPLER_CONFIG.oracle_annotation_path = ''
cfg.DIVERSITY_SAMPLER_CONFIG.dataset_type = ''

cfg.merge_from_file(args.config)

# 动态赋值
cfg.TRAIN_CONFIG = os.path.join(cfg.CONFIG_DIR, 'al_train/retinanet_26e.yaml')
cfg.UNCERTAINTY_INFER_CONFIG = os.path.join(cfg.CONFIG_DIR, 'al_inference/retinanet_uncertainty.yaml')
cfg.DIVERSITY_INFER_CONFIG = os.path.join(cfg.CONFIG_DIR, 'al_inference/retinanet_diversity.yaml')

cfg.OUTPUT_DIR = os.path.join(cfg.WORK_DIR, 'retinanet_coco_ppal_5rounds_2percent_to_10percent')

cfg.UNCERTAINTY_SAMPLER_CONFIG.oracle_annotation_path = cfg.ORACLE_PATH
cfg.DIVERSITY_SAMPLER_CONFIG.n_sample_images = cfg.BUDGET
cfg.DIVERSITY_SAMPLER_CONFIG.oracle_annotation_path = cfg.ORACLE_PATH
cfg.freeze()

sys_echo('>> Start COCO active learning')
sys_echo('>> Working path: %s' % cfg.OUTPUT_DIR)
sys_echo('>> Config: %s' % args.config)
sys_echo('\n')

PYTHON = cfg.PYTHON_PATH

def get_start_round():
    start_round = 0
    if args.resume:
        if not os.path.isdir(cfg.OUTPUT_DIR):
            pass  # output_dir does not exist, starting from scratch
        else:
            k = 0
            while k < cfg.ROUND_NUM:
                round_work_dir = os.path.join(cfg.OUTPUT_DIR, 'round%d' % (k + 1))
                if os.path.isfile(os.path.join(round_work_dir, 'annotations', 'new_labeled.json')):
                    k += 1
                else:
                    break
            start_round = k
    return start_round

# init Al sampler
cfg.defrost()
cfg.UNCERTAINTY_SAMPLER_CONFIG.n_sample_images = \
    cfg.BUDGET * cfg.BUDGET_EXPAND_RATIO + \
        cfg.GPUS - \
            (cfg.BUDGET * cfg.BUDGET_EXPAND_RATIO) % cfg.GPUS
            

cfg.freeze()

uncertainty_sampler = build_al_sampler(cfg.UNCERTAINTY_SAMPLER_CONFIG)
diversity_sampler = build_al_sampler(cfg.DIVERSITY_SAMPLER_CONFIG)

def run(round, run_al):
    last_round_work_dir                     = os.path.join(cfg.OUTPUT_DIR, 'round%d'%(round-1))
    round_work_dir                          = os.path.join(cfg.OUTPUT_DIR, 'round%d'%round)
    round_labeled_json                      = os.path.join(round_work_dir, 'annotations', 'labeled.json')
    round_unlabeled_json                    = os.path.join(round_work_dir, 'annotations', 'unlabeled.json')
    round_eval_log                          = os.path.join(round_work_dir, 'eval.txt')

    round_uncertainty_inference_json_prefix = os.path.join(round_work_dir, 'unlabeled_inference_result')
    round_uncertainty_inference_dir        = os.path.join(round_work_dir, f'uncertainty_coco_active_round_{round}/')      # 推理后结果保存; evaluator内部保存名
    round_uncertainty_new_labeled_json      = os.path.join(round_work_dir, 'annotations', 'uncertainty_new_labeled.json')
    round_uncertainty_new_unlabeled_json    = os.path.join(round_work_dir, 'annotations', 'uncertainty_new_unlabeled.json')

    round_diversity_image_dis_npy           = os.path.join(round_work_dir, 'image_dis.npy')
    round_diversity_inference_json_prefix   = os.path.join(round_work_dir, 'diversity_inference_result')
    round_diversity_inference_json          = os.path.join(round_work_dir, 'diversity_inference_result.bbox.json')
    round_diversity_new_labeled_json        = os.path.join(round_work_dir, 'annotations', 'new_labeled.json')
    round_diversity_new_unlabeled_json      = os.path.join(round_work_dir, 'annotations', 'new_unlabeled.json')

    train_command = 'export MKL_THREADING_LAYER=GNU && ' + \
                    '%s -m torch.distributed.launch '%PYTHON + \
                    ' --nproc_per_node=%d ' % int(cfg.GPUS) + \
                    ' --master_port=%d ' % int(cfg.PORT) + \
                    ' tools/al/train.py ' + \
                    ' --config-file %s ' % cfg.TRAIN_CONFIG + \
                    ' LABELED_DATA %s ' % round_labeled_json + \
                    ' UNLABELED_DATA %s ' % round_unlabeled_json + \
                    ' ROUND_IDX %d ' % round + \
                    ' DATA_JSON %s ' % round_labeled_json + \
                    ' OUTPUT_DIR %s ' % round_work_dir + \
                    ' IMG_ROOT %s ' % cfg.IMG_ROOT + \
                    " MODEL.WEIGHTS ./models/model_final_bfca0b.pkl"
                    
                    # ' MODEL.WEIGHTS ./models/model_final_bfca0b.pkl' + \
                    

    eval_command = 'export MKL_THREADING_LAYER=GNU && ' + \
                   '%s -m torch.distributed.launch '%PYTHON + \
                   ' --nproc_per_node=%d ' % int(cfg.GPUS) + \
                   ' --master_port=%d ' % int(cfg.PORT) + \
                   ' tools/al/test.py ' + \
                   ' --config-file %s ' % cfg.TRAIN_CONFIG + \
                   ' LABELED_DATA %s ' % round_labeled_json + \
                   ' UNLABELED_DATA %s ' % round_unlabeled_json + \
                    ' ROUND_IDX %d ' % round + \
                   ' MODEL.WEIGHTS %s ' % os.path.join(round_work_dir, 'model_final.pth') + \
                   ' OUTPUT_DIR %s ' % round_work_dir + \
                   ' IMG_ROOT %s ' % cfg.IMG_ROOT + \
                    ' INFER False ' + \
                   ' > %s' % round_eval_log

    # 和区别是数据集不同：round_unlabeled_json 
    unlabeled_infer_command = 'export MKL_THREADING_LAYER=GNU && ' + \
                              '%s -m torch.distributed.launch '%PYTHON + \
                              ' --nproc_per_node=%d ' % int(cfg.GPUS) + \
                              ' --master_port=%d ' % int(cfg.PORT) + \
                              ' tools/al/test.py ' + \
                              ' --config-file %s ' % cfg.UNCERTAINTY_INFER_CONFIG + \
                              ' MODEL.WEIGHTS %s ' % os.path.join(round_work_dir, 'model_final.pth') + \
                              ' OUTPUT_DIR %s ' % round_work_dir + \
                              ' ROUND_IDX %d ' % round + \
                              ' IMG_ROOT %s ' % cfg.IMG_ROOT + \
                              ' DATA_JSON %s ' % round_unlabeled_json  + \
                              ' INFER True ' + \
                              ' INFER_DIVERSITY False '
                            #   ' --format-only ' + \
                            #   ' --eval-options \"jsonfile_prefix=%s\"' % round_uncertainty_inference_json_prefix +\
                            #   ' --cfg-options unlabeled_data=%s data.test.ann_file=%s' % (round_unlabeled_json, round_unlabeled_json)

    if args.model == 'fasterrcnn':
        head = 'roi_head'
    else:
        head = 'bbox_head'

    os.system('mkdir -p %s' % os.path.join(round_work_dir, 'annotations'))
    if round == 1:
        os.system('cp %s %s' % (cfg.INIT_LABELED_JSON, round_labeled_json))
        os.system('cp %s %s' % (cfg.INIT_UNLABELED_JSON, round_unlabeled_json))
        if cfg.INIT_MODEL is not None:
            os.system('cp %s %s'%(cfg.INIT_MODEL,os.path.join(round_work_dir, 'model_final.pth')))
        else:
            command_with_time(train_command, 'Training')
        # if cfg.INIT_INFERENCE_RESULTS is not None:
        #     os.system('cp %s %s'%(cfg.INIT_INFERENCE_RESULTS, round_uncertainty_inference_json))
    else:
        if args.resume and os.path.isfile(os.path.join(round_work_dir, 'model_final.pth')) :
            pass
        else:
            os.system('cp %s %s' % (os.path.join(last_round_work_dir, 'annotations', 'new_labeled.json'), round_labeled_json))
            os.system('cp %s %s' % (os.path.join(last_round_work_dir, 'annotations', 'new_unlabeled.json'), round_unlabeled_json))
            command_with_time(train_command, 'Training')

    if not (os.path.isfile(round_eval_log) and args.resume):        
        command_with_time(eval_command, 'Evaluation round %d'%round)
    display_latest_results(cfg.OUTPUT_DIR, round, os.path.join(cfg.OUTPUT_DIR, 'eval_results.txt'))

    if run_al:
        if not (os.path.isfile(round_uncertainty_inference_dir+'/coco_instances_results.json') and args.resume):
            command_with_time(unlabeled_infer_command, 'Inference on unlabeled data')
        if not (os.path.isfile(round_uncertainty_new_labeled_json) and os.path.isfile(round_uncertainty_new_unlabeled_json) and args.resume):
            uncertainty_sampler.al_round(round_work_dir, round_uncertainty_inference_dir, round_labeled_json, round_uncertainty_new_labeled_json, round_uncertainty_new_unlabeled_json)

        if hasattr(uncertainty_sampler, 'get_pool_size'):
            pool_size_round = uncertainty_sampler.get_pool_size(round + 1)
        else:
            pool_size_round = int(uncertainty_sampler.n_images)

        diversity_infer_command = 'export MKL_THREADING_LAYER=GNU && ' + \
                                  '%s -m torch.distributed.launch ' % PYTHON + \
                                  ' --nproc_per_node=%d ' % int(cfg.GPUS) + \
                                  ' --master_port=%d ' % int(cfg.PORT) + \
                                  ' tools/al/test.py ' + \
                                  ' --config-file %s ' % cfg.DIVERSITY_INFER_CONFIG + \
                                  ' MODEL.WEIGHTS %s ' % os.path.join(round_work_dir, 'model_final.pth') + \
                                  ' OUTPUT_DIR %s ' % round_work_dir + \
                                  ' IMG_ROOT %s ' % cfg.IMG_ROOT + \
                                  ' DATA_JSON %s ' % round_uncertainty_new_labeled_json + \
                                  ' INFER True ' + \
                                  ' INFER_DIVERSITY True ' + \
                                  ' ROUND_IDX %d ' % round + \
                                  ' TOTAL_IMAGES %d' % pool_size_round + \
                                  ' OUTPUT_PATH %s ' % round_diversity_image_dis_npy
                                  
                                #   ' --format-only ' + \
                                #   ' --eval-options \"jsonfile_prefix=%s\"' % round_diversity_inference_json_prefix + \
                                #   ' --cfg-options unlabeled_data=%s data.test.ann_file=%s' % (round_uncertainty_new_labeled_json, round_uncertainty_new_labeled_json) + \
        if not (os.path.isfile(round_diversity_image_dis_npy) and args.resume):
            command_with_time(diversity_infer_command, 'Inference on diversity data')
        if not (os.path.isfile(round_diversity_new_labeled_json) and args.resume):
            diversity_sampler.al_round(round_uncertainty_inference_dir, round_diversity_image_dis_npy, round_labeled_json, round_diversity_new_labeled_json, round_diversity_new_unlabeled_json)

        # delete inference results because they are too large
        os.system('rm -f %s' % round_uncertainty_inference_dir)
        os.system('rm -f %s' % round_diversity_inference_json)
        os.system('rm -f %s' % round_diversity_image_dis_npy)


if __name__ == '__main__':
    start_round = get_start_round()
    os.system('mkdir -p %s' % cfg.OUTPUT_DIR)
    os.system('cp %s %s' % (args.config, os.path.join(cfg.OUTPUT_DIR, os.path.split(args.config)[-1])))
    uncertainty_sampler.set_round(start_round+1)
    diversity_sampler.set_round(start_round + 1)
    for i in range(start_round, int(cfg.ROUND_NUM)):
        run(i+1, i!=int(cfg.ROUND_NUM)-1)
    pass
