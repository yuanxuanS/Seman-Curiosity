import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.detectors import SegmentationDetections


COCO_METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def load_and_normalise_coco(annotation_path: Path) -> dict:
    """Load COCO JSON and convert Detectron2 XYXY boxes to standard XYWH."""
    with annotation_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    dataset = copy.deepcopy(dataset)
    dataset.setdefault("info", {})
    dataset.setdefault("licenses", [])
    for annotation in dataset.get("annotations", []):
        if "bbox_mode" not in annotation:
            continue
        # Detectron2 BoxMode.XYXY_ABS == 0. Existing repository exports use it.
        if int(annotation.pop("bbox_mode")) != 0:
            raise ValueError("Only Detectron2 XYXY_ABS (bbox_mode=0) is supported.")
        x1, y1, x2, y2 = annotation["bbox"]
        annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    return dataset


def create_coco_api(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def build_model_to_dataset_category_map(
    model_names: Mapping[int, str],
    categories: Sequence[dict],
) -> Dict[int, int]:
    """Map model class IDs to dataset IDs by category name, not numeric ID."""
    dataset_by_name = {str(item["name"]).strip().lower(): int(item["id"]) for item in categories}
    mapping = {
        int(model_id): dataset_by_name[name.strip().lower()]
        for model_id, name in model_names.items()
        if name.strip().lower() in dataset_by_name
    }
    if not mapping:
        raise ValueError(
            "The model and dataset have no category names in common. "
            f"Model examples: {list(model_names.values())[:5]}; "
            f"dataset: {list(dataset_by_name)}"
        )
    return mapping


def _encode_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def detections_to_coco_results(
    image_id: int,
    detections: SegmentationDetections,
    category_mapping: Mapping[int, int],
) -> Tuple[List[dict], List[dict]]:
    boxes = detections.boxes.detach().cpu().numpy()
    classes = detections.classes.detach().cpu().numpy()
    scores = detections.scores.detach().cpu().numpy()
    masks = detections.masks.detach().cpu().numpy()
    bbox_results, segmentation_results = [], []

    for box, model_class, score, mask in zip(boxes, classes, scores, masks):
        model_class = int(model_class)
        if model_class not in category_mapping:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        common = {
            "image_id": int(image_id),
            "category_id": int(category_mapping[model_class]),
            "score": float(score),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        }
        bbox_results.append(common)
        segmentation_results.append({**common, "segmentation": _encode_mask(mask)})
    return bbox_results, segmentation_results


def evaluate_coco_results(
    coco_gt: COCO,
    results: Sequence[dict],
    iou_type: str,
    image_ids: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    if not results:
        return {name: 0.0 for name in COCO_METRIC_NAMES}
    coco_dt = coco_gt.loadRes(list(results))
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if image_ids is not None:
        evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        name: float(value)
        for name, value in zip(COCO_METRIC_NAMES, evaluator.stats.tolist())
    }


def evaluate_coco_results_per_category(
    coco_gt: COCO,
    results: Sequence[dict],
    iou_type: str,
    image_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run COCO evaluation independently for every dataset category."""
    categories = sorted(
        coco_gt.dataset.get("categories", []), key=lambda category: int(category["id"])
    )
    per_category = {}
    for category in categories:
        category_id = int(category["id"])
        category_results = [
            result for result in results if int(result["category_id"]) == category_id
        ]
        if not category_results:
            metrics = {name: 0.0 for name in COCO_METRIC_NAMES}
        else:
            coco_dt = coco_gt.loadRes(category_results)
            evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            evaluator.params.catIds = [category_id]
            if image_ids is not None:
                evaluator.params.imgIds = list(image_ids)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            metrics = {
                name: float(value)
                for name, value in zip(COCO_METRIC_NAMES, evaluator.stats.tolist())
            }
        per_category[str(category["name"])] = metrics
    return per_category
