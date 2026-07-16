"""Ultralytics YOLO wrapper for metro infrastructure detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from metro_navigation.config import ObjectDetectorSettings
from metro_navigation.domain import ObjectDetection, ObjectDetectionResult, Point
from metro_navigation.utils.clock import anchor_line_geometry, points_to_clock

LOGGER = logging.getLogger(__name__)
ESCALATOR_ENTRY_CLASS = "escalator entry node"


class ObjectDetector:
    """Detect metro infrastructure and attach clock-face directions."""

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str,
        settings: ObjectDetectorSettings,
    ) -> None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Object-detector weights not found: {weights_path}")

        self._device = device
        self._settings = settings
        self._model = YOLO(str(weights_path))
        self._class_names = self._model.names
        self._minimum_predict_confidence = min(
            settings.default_confidence,
            settings.high_confidence,
        )

        try:
            self._model.fuse()
        except Exception:
            LOGGER.debug("YOLO layer fusion was unavailable", exc_info=True)

    def _class_name(self, class_id: int) -> str:
        names: Any = self._class_names
        if isinstance(names, dict):
            return str(names[class_id])
        return str(names[class_id])

    @torch.inference_mode()
    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        root_point: Point | None = None,
    ) -> ObjectDetectionResult:
        """Run object detection on one BGR frame."""

        height, width = frame_bgr.shape[:2]
        if root_point is None:
            _, _, root_point, _ = anchor_line_geometry(width, height)

        result = self._model.predict(
            frame_bgr,
            imgsz=self._settings.image_size,
            device=self._device,
            conf=self._minimum_predict_confidence,
            verbose=False,
        )[0]

        detections: list[ObjectDetection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy()

            for box, confidence_raw, class_id_raw in zip(
                xyxy,
                confidences,
                class_ids,
                strict=True,
            ):
                class_name = self._class_name(int(class_id_raw))
                confidence = float(confidence_raw)
                threshold = (
                    self._settings.high_confidence
                    if class_name in self._settings.high_confidence_classes
                    else self._settings.default_confidence
                )
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = (int(value) for value in box.tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                clock_direction, angle_degrees = points_to_clock(root_point, center)
                detections.append(
                    ObjectDetection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        clock_direction=clock_direction,
                        angle_degrees=angle_degrees,
                    )
                )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        escalators = tuple(
            detection for detection in detections if detection.class_name == ESCALATOR_ENTRY_CLASS
        )
        return ObjectDetectionResult(tuple(detections), escalators)
