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

from src.detectors import YOLOv8SegBackend
from src.evaluation.yolov8_seg_coco import (
    build_model_to_dataset_category_map,
    create_coco_api,
    detections_to_coco_results,
    evaluate_coco_results,
    evaluate_coco_results_per_category,
    load_and_normalise_coco,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 instance segmentation with COCO bbox and segm metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolov8_seg/eval_embodied.yaml"),
    )
    parser.add_argument("--weights", type=str, help="Override model.weights.")
    parser.add_argument("--device", type=str, help="Override model.device, e.g. 0 or cpu.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def batched(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def main():
    args = parse_args()
    config_path = resolve_path(str(args.config))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = dict(config["model"])
    if args.weights:
        model_config["weights"] = args.weights
    if args.device:
        model_config["device"] = args.device

    annotations_path = resolve_path(config["dataset"]["annotations"])
    image_root = resolve_path(config["dataset"]["image_root"])
    output_dir = resolve_path(config["evaluation"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotations_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    dataset = load_and_normalise_coco(annotations_path)
    images = list(dataset["images"])
    limit = args.limit if args.limit is not None else config["evaluation"].get("limit")
    if limit is not None:
        images = images[: int(limit)]
    image_ids = [int(image["id"]) for image in images]

    coco_gt = create_coco_api(dataset)
    backend = YOLOv8SegBackend(**model_config)
    category_mapping = build_model_to_dataset_category_map(
        backend.names, dataset["categories"]
    )
    selected_model_classes = sorted(category_mapping)

    bbox_results, segmentation_results = [], []
    batch_size = int(config["evaluation"]["batch_size"])
    for image_batch in batched(images, batch_size):
        paths = [image_root / image["file_name"] for image in image_batch]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing dataset image(s): {missing[:3]}")
        predictions = backend.predict(paths, classes=selected_model_classes)
        for image, prediction in zip(image_batch, predictions):
            bbox_batch, segmentation_batch = detections_to_coco_results(
                image["id"], prediction, category_mapping
            )
            bbox_results.extend(bbox_batch)
            segmentation_results.extend(segmentation_batch)

    (output_dir / "bbox_predictions.json").write_text(
        json.dumps(bbox_results), encoding="utf-8"
    )
    (output_dir / "segmentation_predictions.json").write_text(
        json.dumps(segmentation_results), encoding="utf-8"
    )

    bbox_metrics = evaluate_coco_results(coco_gt, bbox_results, "bbox", image_ids)
    segmentation_metrics = evaluate_coco_results(
        coco_gt, segmentation_results, "segm", image_ids
    )
    metrics = {
        "model": model_config["weights"],
        "num_images": len(images),
        "categories": category_mapping,
        "bbox": {
            **bbox_metrics,
            "per_category": evaluate_coco_results_per_category(
                coco_gt, bbox_results, "bbox", image_ids
            ),
        },
        "segm": {
            **segmentation_metrics,
            "per_category": evaluate_coco_results_per_category(
                coco_gt, segmentation_results, "segm", image_ids
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
