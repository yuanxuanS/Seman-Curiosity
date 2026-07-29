#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ["SEMANTIC_CURIOSITY_LIGHT_IMPORT"] = "1"

from src.detectors.yolo_dataset import (
    convert_detectron_coco_split,
    write_yolo_data_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg from repository Detectron2/COCO datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolov8_seg/train_embodied.yaml"),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Convert the dataset and stop before model training.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an already prepared YOLO dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Convert at most N images per split; intended only for smoke tests.",
    )
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--batch", type=int, help="Override training.batch.")
    parser.add_argument("--device", type=str, help="Override training.device.")
    parser.add_argument("--weights", type=str, help="Override model.weights.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def prepare_dataset(config: dict, limit=None) -> Path:
    dataset_config = config["dataset"]
    output_root = resolve_path(dataset_config["prepared_root"])
    common = {
        "output_root": output_root,
        "image_mode": dataset_config.get("image_mode", "symlink"),
        "simplify_fraction": float(dataset_config.get("simplify_fraction", 0.001)),
        "minimum_area": float(dataset_config.get("minimum_polygon_area", 4.0)),
        "limit": limit,
    }
    category_names = None
    preparation_stats = {}
    for split in ("train", "val"):
        split_config = dataset_config[split]
        category_names, stats = convert_detectron_coco_split(
            annotation_path=resolve_path(split_config["annotations"]),
            image_root=resolve_path(split_config["image_root"]),
            split=split,
            category_names=category_names,
            **common,
        )
        preparation_stats[split] = stats

    data_yaml = write_yolo_data_yaml(output_root, category_names)
    (output_root / "conversion_stats.json").write_text(
        json.dumps(preparation_stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(preparation_stats, indent=2))
    print(f"YOLO dataset configuration: {data_yaml}")
    return data_yaml


def main():
    args = parse_args()
    config_path = resolve_path(str(args.config))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_root = resolve_path(config["dataset"]["prepared_root"])
    data_yaml = output_root / "data.yaml"
    if not args.skip_prepare:
        data_yaml = prepare_dataset(config, limit=args.limit)
    elif not data_yaml.is_file():
        raise FileNotFoundError(
            f"Prepared data configuration not found: {data_yaml}. "
            "Run without --skip-prepare first."
        )

    if args.prepare_only:
        return

    from ultralytics import YOLO

    weights = args.weights or config["model"]["weights"]
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch is not None:
        training["batch"] = args.batch
    if args.device is not None:
        training["device"] = args.device
    training["project"] = str(resolve_path(training["project"]))

    model = YOLO(str(resolve_path(weights)) if Path(weights).is_file() else weights)
    if not config.get("logging", {}).get("wandb", False):
        # Ultralytics 8.0.135 enables W&B whenever the package is installed.
        # Besides being unnecessary for local training, recent W&B rejects a
        # filesystem path passed through Ultralytics' legacy `project` field.
        from ultralytics.yolo.utils.callbacks import wb

        wb.callbacks.clear()
    model.train(data=str(data_yaml), **training)


if __name__ == "__main__":
    main()
