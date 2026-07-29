import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils


def _decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        mask = mask_utils.decode(segmentation)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        return mask.astype(np.uint8)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return (mask_utils.decode(mask_utils.merge(rles)) > 0).astype(np.uint8)
    raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation).__name__}")


def _mask_to_largest_polygon(
    mask: np.ndarray,
    simplify_fraction: float,
    minimum_area: float,
) -> Tuple[Optional[np.ndarray], int]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= minimum_area
    ]
    if not valid_contours:
        return None, 0

    # YOLO segmentation stores one polygon per instance. Repository masks are
    # normally connected; for a disconnected mask, preserve its largest region
    # instead of turning separate regions into multiple object instances.
    contour = max(valid_contours, key=cv2.contourArea)
    epsilon = simplify_fraction * cv2.arcLength(contour, closed=True)
    polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
    if len(polygon) < 3:
        return None, len(valid_contours)
    return polygon.astype(np.float64), len(valid_contours)


def _safe_relative_image_path(file_name: str) -> Path:
    relative_path = Path(file_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe COCO file_name: {file_name}")
    return relative_path


def _materialise_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(
                f"Prepared image points to a different source: {destination}"
            )
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("image_mode must be one of: symlink, hardlink, copy")


def convert_detectron_coco_split(
    annotation_path: Path,
    image_root: Path,
    output_root: Path,
    split: str,
    category_names: Optional[Sequence[str]] = None,
    image_mode: str = "symlink",
    simplify_fraction: float = 0.001,
    minimum_area: float = 4.0,
    limit: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Convert a Detectron2-compatible COCO instance dataset to YOLO-seg."""
    with annotation_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    categories = sorted(dataset["categories"], key=lambda item: int(item["id"]))
    names = [str(category["name"]) for category in categories]
    if category_names is not None and list(category_names) != names:
        raise ValueError(
            f"Category mismatch for {split}: expected {list(category_names)}, got {names}"
        )
    category_to_contiguous = {
        int(category["id"]): index for index, category in enumerate(categories)
    }

    annotations_by_image = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    images = list(dataset["images"])
    if limit is not None:
        images = images[: int(limit)]

    image_output = output_root / "images" / split
    label_output = output_root / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    stats = {
        "images": 0,
        "instances": 0,
        "written_instances": 0,
        "skipped_crowd": 0,
        "skipped_invalid_mask": 0,
        "disconnected_masks": 0,
    }

    for image in images:
        relative_image = _safe_relative_image_path(str(image["file_name"]))
        source_image = image_root / relative_image
        if not source_image.is_file():
            raise FileNotFoundError(f"Dataset image does not exist: {source_image}")
        destination_image = image_output / relative_image
        _materialise_image(source_image, destination_image, image_mode)

        height, width = int(image["height"]), int(image["width"])
        label_lines = []
        for annotation in annotations_by_image[int(image["id"])]:
            stats["instances"] += 1
            if int(annotation.get("iscrowd", 0)):
                stats["skipped_crowd"] += 1
                continue
            segmentation = annotation.get("segmentation")
            if not segmentation:
                stats["skipped_invalid_mask"] += 1
                continue
            mask = _decode_segmentation(segmentation, height, width)
            polygon, component_count = _mask_to_largest_polygon(
                mask, simplify_fraction, minimum_area
            )
            if polygon is None:
                stats["skipped_invalid_mask"] += 1
                continue
            if component_count > 1:
                stats["disconnected_masks"] += 1

            polygon[:, 0] = np.clip(polygon[:, 0] / width, 0.0, 1.0)
            polygon[:, 1] = np.clip(polygon[:, 1] / height, 0.0, 1.0)
            class_id = category_to_contiguous[int(annotation["category_id"])]
            coordinates = " ".join(f"{value:.8g}" for value in polygon.reshape(-1))
            label_lines.append(f"{class_id} {coordinates}")
            stats["written_instances"] += 1

        label_path = (label_output / relative_image).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        stats["images"] += 1

    # Ultralytics caches the discovered labels beside the split directory.
    # Invalidate it whenever conversion runs, especially when a smoke-test
    # `limit` is followed by full dataset preparation.
    cache_path = label_output.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()
    return names, stats


def write_yolo_data_yaml(output_root: Path, category_names: Sequence[str]) -> Path:
    data_path = output_root / "data.yaml"
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(category_names)},
    }
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path
