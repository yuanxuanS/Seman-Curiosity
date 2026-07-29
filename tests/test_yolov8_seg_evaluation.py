import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


os.environ["SEMANTIC_CURIOSITY_LIGHT_IMPORT"] = "1"

from src.detectors import SegmentationDetections
from src.policy_rl.agents.utils.semantic_prediction import SemanticPredYOLOv8
from src.evaluation.yolov8_seg_coco import (
    build_model_to_dataset_category_map,
    detections_to_coco_results,
    create_coco_api,
    evaluate_coco_results_per_category,
    load_and_normalise_coco,
)


ANNOTATIONS = Path(
    "third_parties/detectron2/datasets/embodied_test/"
    "annotations/instances_test.json"
)


def test_repository_coco_boxes_and_categories_are_normalised():
    dataset = load_and_normalise_coco(ANNOTATIONS)

    assert dataset["annotations"][0]["bbox"] == [0.0, 318.0, 556.0, 322.0]
    assert "bbox_mode" not in dataset["annotations"][0]
    assert build_model_to_dataset_category_map(
        {
            56: "chair",
            57: "couch",
            59: "bed",
            61: "toilet",
            72: "refrigerator",
        },
        dataset["categories"],
    ) == {56: 0, 57: 1, 59: 2, 61: 3, 72: 4}


def test_prediction_conversion_produces_coco_xywh_and_rle():
    detections = SegmentationDetections(
        boxes=torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        classes=torch.tensor([56]),
        scores=torch.tensor([0.9]),
        masks=torch.ones((1, 32, 32), dtype=torch.bool),
        image_size=(32, 32),
    )

    bbox, segmentation = detections_to_coco_results(
        image_id=7,
        detections=detections,
        category_mapping={56: 0},
    )

    assert bbox[0]["bbox"] == [1.0, 2.0, 10.0, 20.0]
    assert bbox[0]["category_id"] == 0
    assert segmentation[0]["segmentation"]["size"] == [32, 32]
    assert isinstance(segmentation[0]["segmentation"]["counts"], str)


def test_detectron2_adapter_preserves_policy_fields():
    detections = SegmentationDetections(
        boxes=torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        classes=torch.tensor([56]),
        scores=torch.tensor([0.9]),
        masks=torch.ones((1, 32, 32), dtype=torch.bool),
        image_size=(32, 32),
    )

    instances = detections.to_detectron2_instances()

    assert instances.image_size == (32, 32)
    assert instances.has("pred_boxes")
    assert instances.has("pred_classes")
    assert instances.has("scores")
    assert instances.has("pred_masks")


def test_sequence_yolov8_adapter_remaps_custom_classes_to_coco_ids():
    adapter = SemanticPredYOLOv8.__new__(SemanticPredYOLOv8)
    adapter.args = SimpleNamespace(visualize=0)
    adapter._model_to_coco = adapter._build_model_to_coco_mapping(
        {0: "chair", 1: "couch", 2: "bed", 3: "toilet", 4: "refrigerator"}
    )
    detections = SegmentationDetections(
        boxes=torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        classes=torch.tensor([2]),
        scores=torch.tensor([0.9]),
        masks=torch.ones((1, 32, 32), dtype=torch.bool),
        image_size=(32, 32),
    )

    semantic, _, instances = adapter._convert_detection(
        detections, np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert instances.pred_classes.tolist() == [59]
    assert semantic.shape == (32, 32, 6)
    assert semantic[:, :, 3].sum() == 32 * 32


def test_empty_per_category_metrics_include_every_category():
    dataset = load_and_normalise_coco(ANNOTATIONS)
    metrics = evaluate_coco_results_per_category(
        create_coco_api(dataset),
        results=[],
        iou_type="bbox",
        image_ids=[dataset["images"][0]["id"]],
    )

    assert list(metrics) == ["chair", "couch", "bed", "toilet", "refrigerator"]
    assert all(category_metrics["AP"] == 0.0 for category_metrics in metrics.values())
