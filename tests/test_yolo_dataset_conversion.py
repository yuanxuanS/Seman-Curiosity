import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


os.environ["SEMANTIC_CURIOSITY_LIGHT_IMPORT"] = "1"

from src.detectors.yolo_dataset import (
    convert_detectron_coco_split,
    write_yolo_data_yaml,
)


def _rle(mask):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def test_detectron_coco_rle_converts_to_yolo_polygon(tmp_path):
    image_root = tmp_path / "source"
    image_root.mkdir()
    cv2.imwrite(str(image_root / "sample.png"), np.zeros((32, 32, 3), np.uint8))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:20, 6:24] = 1
    annotations = {
        "images": [
            {"id": 1, "file_name": "sample.png", "height": 32, "width": 32}
        ],
        "categories": [{"id": 9, "name": "chair"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 9,
                "segmentation": _rle(mask),
                "iscrowd": 0,
            }
        ],
    }
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    output_root = tmp_path / "prepared"

    names, stats = convert_detectron_coco_split(
        annotation_path,
        image_root,
        output_root,
        split="train",
    )
    data_yaml = write_yolo_data_yaml(output_root, names)

    values = (output_root / "labels/train/sample.txt").read_text().split()
    assert names == ["chair"]
    assert stats["written_instances"] == 1
    assert values[0] == "0"
    assert len(values) >= 7
    assert all(0.0 <= float(value) <= 1.0 for value in values[1:])
    assert (output_root / "images/train/sample.png").is_symlink()
    assert data_yaml.is_file()
