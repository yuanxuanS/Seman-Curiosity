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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an inference visualization for every evaluated image.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help="Visualization output directory (default: <evaluation.output>/visualizations).",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def batched(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def category_color(category_id: int):
    """Return a stable, visually distinct BGR color for a dataset category."""
    palette = (
        (60, 20, 220),
        (230, 0, 0),
        (45, 108, 142),
        (30, 170, 100),
        (32, 11, 119),
        (0, 165, 255),
        (180, 105, 255),
        (128, 128, 0),
    )
    return palette[int(category_id) % len(palette)]


def save_prediction_visualization(
    image_path,
    output_path,
    prediction,
    category_mapping,
    category_names,
    mask_alpha=0.45,
):
    """Draw masks, boxes, class names and scores on one test image."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read image for visualization: {image_path}")

    boxes = prediction.boxes.detach().cpu().numpy()
    classes = prediction.classes.detach().cpu().numpy()
    scores = prediction.scores.detach().cpu().numpy()
    masks = prediction.masks.detach().cpu().numpy().astype(bool)

    overlay = image.copy()
    visible = []
    for box, model_class, score, mask in zip(boxes, classes, scores, masks):
        model_class = int(model_class)
        if model_class not in category_mapping:
            continue
        category_id = int(category_mapping[model_class])
        color = category_color(category_id)
        overlay[mask] = color
        visible.append((box, category_id, float(score), color))

    image = cv2.addWeighted(overlay, mask_alpha, image, 1.0 - mask_alpha, 0.0)
    font_scale = 0.8
    font_thickness = 2
    text_padding = 6
    for box, category_id, score, color in visible:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{category_names[category_id]} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_top = max(0, y1 - text_height - baseline - 2 * text_padding)
        cv2.rectangle(
            image,
            (x1, text_top),
            (
                x1 + text_width + 2 * text_padding,
                text_top + text_height + baseline + 2 * text_padding,
            ),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + text_padding, text_top + text_height + text_padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save visualization: {output_path}")


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
    visualize = args.visualize or bool(config["evaluation"].get("visualize", False))
    visualization_dir = (
        resolve_path(str(args.visualization_dir))
        if args.visualization_dir is not None
        else output_dir / config["evaluation"].get("visualization_dir", "visualizations")
    )
    if visualize:
        visualization_dir.mkdir(parents=True, exist_ok=True)

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
    category_names = {
        int(category["id"]): str(category["name"])
        for category in dataset["categories"]
    }
    selected_model_classes = sorted(category_mapping)

    bbox_results, segmentation_results = [], []
    batch_size = int(config["evaluation"]["batch_size"])
    for image_batch in batched(images, batch_size):
        paths = [image_root / image["file_name"] for image in image_batch]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing dataset image(s): {missing[:3]}")
        predictions = backend.predict(paths, classes=selected_model_classes)
        for image, path, prediction in zip(image_batch, paths, predictions):
            bbox_batch, segmentation_batch = detections_to_coco_results(
                image["id"], prediction, category_mapping
            )
            bbox_results.extend(bbox_batch)
            segmentation_results.extend(segmentation_batch)
            if visualize:
                save_prediction_visualization(
                    path,
                    visualization_dir / image["file_name"],
                    prediction,
                    category_mapping,
                    category_names,
                )

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
        "visualization_dir": str(visualization_dir) if visualize else None,
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
