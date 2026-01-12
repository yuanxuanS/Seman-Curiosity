import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import hydra
import numpy as np
from src.finetune.sensors_data import AgentPoseSense, BBSense
from src.finetune.dataset_utils import get_loader, SampleLoader
import tqdm
import torch.nn as nn
# pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps_finetune/2025-02-21/02-21-10-44-16_gt_consensus_vanilla_bs_8_notes_zone/checkpoints/epoch=09.ckpt"
# data = torch.load(pth)
# print(data.keys())
# print(data['state_dict']['student_model.feature_projector.layer1.0.bias'])

# pth_2 = '/home/users/wpp/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
# data2 = torch.load(pth_2)
# # print(data2.keys())
# print(data2['state_dict']['student_model.feature_projector.layer1.0.bias'])

# pth_pretrained = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps_finetune/2025-04-08/04-08-09-50-27_visualize_lbs_bs_8_notes_visualize/iteration-0.ckpt"
# data3 = torch.load(pth_pretrained)
# print(data3.keys())
# print(['feature_projector.layer1.0.bias'])

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg

def reinit_head(model, classes_idxs): 
    model.roi_heads.num_classes = len(classes_idxs)

    # if isinstance(self.model.roi_heads.box_predictor, MinimalPredictorWrapper):
    #     self.model.roi_heads.box_predictor.reinit_head(classes_idxs)
    # else:
    classes_to_keep = np.array([*classes_idxs, 80])
    cls_bias = torch.nn.Parameter(
        model.roi_heads.box_predictor.cls_score.bias[classes_to_keep]
    )
    cls_weight = torch.nn.Parameter(
        model.roi_heads.box_predictor.cls_score.weight[classes_to_keep]
    )

    classes_to_keep = np.array([*classes_idxs])
    mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
        np.arange(0, 4), len(classes_to_keep)
    )

    box_weight = torch.nn.Parameter(
        model.roi_heads.box_predictor.bbox_pred.weight[mask]
    )
    box_bias = torch.nn.Parameter(
        model.roi_heads.box_predictor.bbox_pred.bias[mask]
    )
    model.roi_heads.box_predictor.num_classes = len(classes_idxs)

    in_features = box_weight.shape[1]
    model.roi_heads.box_predictor.cls_score = nn.Linear(
        in_features, len(classes_to_keep) + 1
    )
    model.roi_heads.box_predictor.cls_score.bias = cls_bias
    model.roi_heads.box_predictor.cls_score.weight = cls_weight

    model.roi_heads.box_predictor.bbox_pred = nn.Linear(
        in_features, len(classes_idxs) * 4
    )
    model.roi_heads.box_predictor.bbox_pred.bias = box_bias
    model.roi_heads.box_predictor.bbox_pred.weight = box_weight

    if hasattr(model.roi_heads, "mask_head"):
        classes_to_keep = np.array([*classes_idxs])
        mask_weight = torch.nn.Parameter(
            model.roi_heads.mask_head.predictor.weight[classes_to_keep]
        )
        mask_bias = torch.nn.Parameter(
            model.roi_heads.mask_head.predictor.bias[classes_to_keep]
        )
        model.roi_heads.mask_head.predictor.weight = mask_weight
        model.roi_heads.mask_head.predictor.bias = mask_bias
        model.roi_heads.mask_head.predictor.num_classes = len(classes_idxs)
    return model

def save_image(self, img, instance: Instances, idx: int, img_pth):
        v = Visualizer(
                img, MetadataCatalog.get('coco_2017_val'))
        cls_id_map = {0: 56, 1:57, 2:58, 3:59, 4:61}
        instances_mapped = map_to_original_cls_gt(instance.to("cpu"), cls_id_map)
        v = v.draw_instance_gt(instances_mapped)
        # img = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_pth + "/rcnn_imgs/img_"+str(idx)+".png", v.get_image())
        
@hydra.main(config_path='./configs_finetune/', config_name='visualize.yaml')
def main(cfg):
    cfg = setup_cfg(cfg.detectron_args)
    model = build_model(cfg)
    # 剪切目标类别
    model = reinit_head(model, BBSense.CLASSES)
    
    model.eval()
    # print(model.state_dict().keys())
    # print(model.state_dict()['roi_heads.mask_head.predictor.weight'][0, :20, :, :])
    
    pth_pretrained = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps_finetune/2025-04-08/04-08-15-08-56_gt_consensus_vanilla_bs_8_notes_obnsV3/detector.pth"
    # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps_finetune/2025-02-21/02-21-10-44-16_gt_consensus_vanilla_bs_8_notes_zone/checkpoints/epoch=09.ckpt"
    data = torch.load(pth_pretrained)
    print(data['state_dict'].keys())
    for k, v in model.state_dict().items():
        layer_name = 'student_model.model.' + k
        if layer_name in data['state_dict'].keys():
            model.state_dict()[k].copy_(data['state_dict'][layer_name])
            print(f"layer is {layer_name}")
    print(model.state_dict()['roi_heads.mask_head.predictor.weight'][0, :20, :, :])
    
    
    base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_v2_eval_best_sample/"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_eval_best_sample/"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/expv7_eval_best2/"

    dataset_path = base_dir + "/episodes_data"
    env, epi, step = 0,1,5
    sampler = SampleLoader(dataset_path)

    # construct dataset
    transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels', 'infos'],
            ),
        )

        
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
    
    # dataloader wrap
    test_loader = get_loader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=dict_helper_collate,
        )

    # inference
    n = 0
    for batch in tqdm.tqdm(test_loader):
        # images = batch["image"]
        pred = model(batch)[0]
        # visualize
        for y, x in zip(pred, batch):
            instance = y['instances'].to('cpu')
            image = x['image'].permute()
            
            save_image(image, instance, n, "./")
            n += 1
if __name__ == '__main__':
    main()