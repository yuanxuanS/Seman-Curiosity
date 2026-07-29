from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


ImageInput = Union[str, Path, np.ndarray]


@dataclass
class SegmentationDetections:
    """Model-independent instance-segmentation output in original image size."""

    boxes: torch.Tensor
    classes: torch.Tensor
    scores: torch.Tensor
    masks: torch.Tensor
    image_size: tuple

    def to(self, device: Union[str, torch.device]) -> "SegmentationDetections":
        return SegmentationDetections(
            boxes=self.boxes.to(device),
            classes=self.classes.to(device),
            scores=self.scores.to(device),
            masks=self.masks.to(device),
            image_size=self.image_size,
        )

    def to_detectron2_instances(self):
        """Convert to the structure consumed by the current policy agents."""
        try:
            from detectron2.structures import Boxes, Instances
        except ImportError as exc:
            raise ImportError(
                "Detectron2 is required only for converting YOLO output to Instances."
            ) from exc

        return Instances(
            image_size=self.image_size,
            pred_boxes=Boxes(self.boxes),
            pred_classes=self.classes,
            scores=self.scores,
            pred_masks=self.masks,
        )


class YOLOv8SegBackend:
    """Thin Ultralytics adapter returning stable, repository-native outputs."""

    def __init__(
        self,
        weights: Union[str, Path] = "yolov8n-seg.pt",
        device: Optional[Union[str, int]] = None,
        confidence: float = 0.001,
        iou: float = 0.7,
        image_size: int = 640,
        max_detections: int = 300,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run this repository in the 'explore' "
                "environment or install the pinned dependency from requirements.txt."
            ) from exc

        self.weights = str(weights)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max_detections
        self.model = YOLO(self.weights)
        self.names: Dict[int, str] = self._normalise_names(self.model.names)

    @staticmethod
    def _normalise_names(names: Union[Mapping, Sequence]) -> Dict[int, str]:
        if isinstance(names, Mapping):
            return {int(key): str(value) for key, value in names.items()}
        return {idx: str(value) for idx, value in enumerate(names)}

    @torch.inference_mode()
    def predict(
        self,
        images: Union[ImageInput, Sequence[ImageInput]],
        classes: Optional[Sequence[int]] = None,
    ) -> List[SegmentationDetections]:
        if isinstance(images, (str, Path, np.ndarray)):
            images = [images]

        results = self.model.predict(
            source=list(images),
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            max_det=self.max_detections,
            classes=None if classes is None else list(classes),
            device=self.device,
            retina_masks=True,
            verbose=False,
            stream=False,
        )
        return [self._convert_result(result) for result in results]

    @staticmethod
    def _convert_result(result) -> SegmentationDetections:
        height, width = (int(result.orig_shape[0]), int(result.orig_shape[1]))
        device = result.boxes.xyxy.device

        boxes = result.boxes.xyxy.detach()
        classes = result.boxes.cls.detach().to(dtype=torch.int64)
        scores = result.boxes.conf.detach()

        if result.masks is None or len(boxes) == 0:
            masks = torch.zeros((len(boxes), height, width), dtype=torch.bool, device=device)
        else:
            masks = result.masks.data.detach()
            if tuple(masks.shape[-2:]) != (height, width):
                masks = F.interpolate(
                    masks[:, None].float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            masks = masks > 0.5

        return SegmentationDetections(
            boxes=boxes,
            classes=classes,
            scores=scores,
            masks=masks,
            image_size=(height, width),
        )
