"""Detector backends shared by policy inference and offline evaluation."""

from .yolov8_seg import SegmentationDetections, YOLOv8SegBackend

__all__ = ["SegmentationDetections", "YOLOv8SegBackend"]
