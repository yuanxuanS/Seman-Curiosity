from src.finetune.detector.augmentations import get_transform
import albumentations as A
from src.finetune.dataset_habitat import BbsgtDataset
from src.finetune.dataset_utils import get_loader, SampleLoader
from src.finetune.utils.train_helpers import dict_helper_collate
from src.finetune.sensors_data import AgentPoseSense, BBSense
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from copy import deepcopy
import torch
from torch import Tensor
import cv2
import os

def save_data_imgs(dataset_path, save_pth):
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    transform = A.Compose(
                get_transform("none"),
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['class_labels', 'infos'],
                ),
            )

    sampler = SampleLoader(dataset_path)
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
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=dict_helper_collate,
                )

    for batch_idx, batch in enumerate(test_loader):
        for idx, x in enumerate(batch):
            remap = BBSense.REMAP
            metadata = MetadataCatalog.get('coco_2017_val')
            visualizer = Visualizer(
                deepcopy(x['image'].permute(1, 2, 0).cpu()),
                metadata,
                instance_mode=ColorMode.IMAGE,
            )
            y = deepcopy(x['instances'])

            y.pred_classes = torch.tensor([remap[p.item()] for p in y.gt_classes])
            if hasattr(y, "gt_boxes"):
                y.pred_boxes = y.gt_boxes
            if hasattr(y, "gt_masks"):
                if isinstance(y.gt_masks, Tensor):
                    y.pred_masks = y.gt_masks
                else:
                    y.pred_masks = y.gt_masks.tensor
            frame = visualizer.draw_instance_predictions(
                predictions=y.to('cpu')
            ).get_image()
            cv2.imwrite(save_pth + "/batch_"+str(batch_idx)+"_img_"+str(idx)+".png", frame)
            
    # break

def play_imgs(path):
    print(f"img in {path}")
    for img in sorted(os.listdir(path)):
        pth = path + "/" + img
        data = cv2.imread(pth)
        cv2.imshow("dataset", data)
        cv2.waitKey(100)
    
if __name__ == "__main__":   
    base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_v3_eval_best_2_sample/"
    # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/expv7_eval_best2/"
    save_pth = base_dir + "episodes_data_imgs"
    dataset_path = base_dir + "episodes_data"

    save_data_imgs(dataset_path, save_pth)
    
    # play_imgs(save_pth)