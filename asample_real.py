import argparse
from pathlib import Path

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from src.vqf_constants import target_coco_categories_mapping
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes, BoxMode

def build_predictor(config_path: Path, model_path: Path, score_thresh: float, device: str) -> DefaultPredictor:
    cfg = get_cfg()
    # Keep custom keys to be compatible with existing embodied config.
    cfg.VIS = False
    cfg.SAVE_PTH = ""
    cfg.DATASET_NAME = ""
    cfg.MODEL.NUM_CLASSES = 80
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def collect_images(image_dir: Path) -> list:
    # return sorted(
    #     p for p in image_dir.glob("*.jpg") if "move" in p.name
    # )
    return sorted(
        p for p in image_dir.glob("*.png")
    )


def main() -> None:
    parser = argparse.ArgumentParser("Batch detectron2 inference for real_data move images")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("real_data/traj_0528_4"),
        help="Directory containing input jpg images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("third_parties/detectron2/configs/embodied/mask_rcnn_R_50_FPN_1x_embodied.yaml"),
        help="Path to detectron2 config file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("third_parties/detectron2/models/model_final_a54504.pkl"),
        help="Path to model weights",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="ROI head score threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="move_instances.pth",
        help="Saved filename under image-dir",
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Whether to remap non-target categories to empty instance (useful for CLIP feature extraction)",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    config_path = args.config.resolve()
    model_path = args.model.resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No jpg image with 'move' in name found under: {image_dir}")

    predictor = build_predictor(config_path, model_path, args.score_thresh, args.device)

    result_dict = {}
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skip invalid image: {image_path}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        if args.remap:
            
            if len(instances) > 0:
                new_instance = Instances(
                            pred_boxes=Boxes(torch.Tensor()),
                            image_size=instances[0].image_size,
                            pred_classes=torch.Tensor(),
                            pred_masks=torch.Tensor(),
                            scores=torch.Tensor(),
                        )
                # for ins in range(len(instances)):
                #     istarget = instances[ins].pred_classes in list(target_coco_categories_mapping.keys())
                #     if istarget:
                
                classes_ = instances.pred_classes
                for i in range(len(classes_)):
                    class_idx = classes_[i]
                    if class_idx in list(target_coco_categories_mapping.keys()):
                        new_instance = new_instance.cat([instances[i]])
                    else:
                        continue
                instances = new_instance           
        result_dict[image_path.name] = instances
        print(f"Done: {image_path.name}")

    save_path = image_dir / args.save_name
    torch.save(result_dict, save_path)
    print(f"Saved {len(result_dict)} entries to: {save_path}")


if __name__ == "__main__":
    main()
