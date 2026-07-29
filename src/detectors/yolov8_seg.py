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

    @staticmethod
    def _select_existing_cuda_device(
        device="", batch=0, newline=False, verbose=True
    ) -> torch.device:
        """Select a CUDA device without changing CUDA_VISIBLE_DEVICES.

        Ultralytics 8.0.x normally rewrites CUDA_VISIBLE_DEVICES and returns
        cuda:0. Habitat-Sim has already initialized CUDA in sequence workers,
        so that late remapping sends every detector to physical GPU 0. Use the
        already assigned physical CUDA index directly instead.
        """
        if isinstance(device, torch.device):
            return device
        value = str(device).lower().strip()
        for token in ("cuda:", "(", ")", "[", "]", "'", " "):
            value = value.replace(token, "")
        if value in ("", "none"):
            return torch.device(
                "cuda:{}".format(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            )
        if value in ("cpu", "mps"):
            return torch.device(value)
        if "," in value:
            raise ValueError(
                "YOLOv8SegBackend expects one GPU per environment worker, "
                "got device={!r}".format(device)
            )
        cuda_index = int(value)
        if not torch.cuda.is_available():
            raise ValueError("CUDA device {} requested but CUDA is unavailable".format(cuda_index))
        if cuda_index >= torch.cuda.device_count():
            raise ValueError(
                "CUDA device {} requested, but only {} devices are visible".format(
                    cuda_index, torch.cuda.device_count()
                )
            )
        return torch.device("cuda:{}".format(cuda_index))

    def __init__(
        self,
        weights: Union[str, Path] = "yolov8n-seg.pt",
        device: Optional[Union[str, int]] = None,
        confidence: float = 0.001,
        iou: float = 0.7,
        image_size: int = 640,
        max_detections: int = 300,
    ):
        # Habitat workers can already have CUDA initialized while their
        # process-wide current device is still cuda:0. Ultralytics performs
        # some CUDA setup without an explicit device during import/predictor
        # initialization. Select the worker's assigned GPU first so those
        # allocations do not create an extra context on physical GPU 0.
        device_value = str(device).lower().replace("cuda:", "").strip()
        if device_value not in ("", "none", "cpu", "mps"):
            if "," in device_value:
                raise ValueError(
                    "YOLOv8SegBackend expects one GPU per environment worker, "
                    "got device={!r}".format(device)
                )
            torch.cuda.set_device(int(device_value))

        try:
            from ultralytics import YOLO
            from ultralytics.yolo.engine import predictor as yolo_predictor
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run this repository in the 'explore' "
                "environment or install the pinned dependency from requirements.txt."
            ) from exc

        # ``BasePredictor.setup_model`` resolves this module-level symbol at
        # inference time. The override is process-local because every
        # environment runs in its own worker process.
        yolo_predictor.select_device = self._select_existing_cuda_device

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
