"""YOLO-based sign detector and spatial sign-pair inference."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from metro_navigation.config import SignDetectorSettings
from metro_navigation.domain import (
    SignComponent,
    SignDetectionResult,
    SignPair,
    SignPlatform,
)

LOGGER = logging.getLogger(__name__)
PLATFORM_CLASS_NAMES = frozenset({"platform-sign", "platform_sign", "platform sign"})


class SignDetector:
    """Detect station signs and infer useful left/right label pairs."""

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str,
        settings: SignDetectorSettings,
    ) -> None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Sign-detector weights not found: {weights_path}")

        self._device = device
        self._settings = settings
        self._model = YOLO(str(weights_path))
        self._class_names = self._model.names
        self._platform_class_id = self._find_platform_class_id()

        try:
            self._model.fuse()
        except Exception:
            LOGGER.debug("YOLO sign-detector layer fusion was unavailable", exc_info=True)

        if self._platform_class_id is None:
            LOGGER.warning(
                "No platform-sign class was found; all sign components will be "
                "grouped into a frame-wide virtual platform"
            )

    def _class_name(self, class_id: int) -> str:
        names: Any = self._class_names
        if isinstance(names, dict):
            return str(names[class_id])
        return str(names[class_id])

    def _find_platform_class_id(self) -> int | None:
        items: Iterable[tuple[int, str]]
        if isinstance(self._class_names, dict):
            items = ((int(key), str(value)) for key, value in self._class_names.items())
        else:
            items = enumerate(str(value) for value in self._class_names)

        for class_id, class_name in items:
            if class_name in PLATFORM_CLASS_NAMES:
                return class_id
        return None

    def _group_by_platform(self, boxes: Any, frame_shape: tuple[int, ...]) -> list[SignPlatform]:
        height, width = frame_shape[:2]
        box_list = list(boxes) if boxes is not None else []

        platform_boxes: list[tuple[float, float, float, float]] = []
        for box in box_list:
            class_id = int(box.cls[0])
            if self._platform_class_id is None or class_id != self._platform_class_id:
                continue
            confidence = float(box.conf[0])
            if confidence < self._settings.confidence:
                continue
            raw_box = box.xyxy[0].tolist()
            platform_boxes.append(
                (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
            )

        platforms = [
            SignPlatform(
                bbox=(
                    int(platform_box[0]),
                    int(platform_box[1]),
                    int(platform_box[2]),
                    int(platform_box[3]),
                )
            )
            for platform_box in platform_boxes
        ]

        def contains(
            platform_box: tuple[float, float, float, float],
            center_x: float,
            center_y: float,
        ) -> bool:
            x1, y1, x2, y2 = platform_box
            return x1 <= center_x <= x2 and y1 <= center_y <= y2

        def nearest_platform_index(center_x: float, center_y: float) -> int | None:
            if not platform_boxes:
                return None
            return min(
                range(len(platform_boxes)),
                key=lambda index: (
                    (center_x - (platform_boxes[index][0] + platform_boxes[index][2]) / 2.0) ** 2
                    + (center_y - (platform_boxes[index][1] + platform_boxes[index][3]) / 2.0) ** 2
                ),
            )

        for box in box_list:
            confidence = float(box.conf[0])
            if confidence < self._settings.confidence:
                continue

            class_id = int(box.cls[0])
            if self._platform_class_id is not None and class_id == self._platform_class_id:
                continue

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            component = SignComponent(
                class_name=self._class_name(class_id),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
            )

            if platform_boxes:
                platform_index = next(
                    (
                        index
                        for index, platform_box in enumerate(platform_boxes)
                        if contains(platform_box, center_x, center_y)
                    ),
                    None,
                )
                if platform_index is None:
                    platform_index = nearest_platform_index(center_x, center_y)
                if platform_index is not None:
                    platforms[platform_index].components.append(component)
            elif not platforms:
                platforms.append(
                    SignPlatform(
                        bbox=(0, 0, width - 1, height - 1),
                        components=[component],
                    )
                )
            else:
                platforms[0].components.append(component)

        for platform in platforms:
            platform.pairs = self._build_pairs(platform)
        return platforms

    def _build_pairs(self, platform: SignPlatform) -> list[SignPair]:
        if not platform.components:
            return []

        _, y1, _, y2 = platform.bbox
        maximum_vertical_offset = self._settings.pair_max_vertical_offset_ratio * max(1, y2 - y1)
        best_by_class: dict[str, SignComponent] = {}
        for component in platform.components:
            previous = best_by_class.get(component.class_name)
            if previous is None or component.confidence > previous.confidence:
                best_by_class[component.class_name] = component

        pairs: list[SignPair] = []

        def append_pair(first_name: str, second_name: str, pair_type: str) -> None:
            first = best_by_class.get(first_name)
            second = best_by_class.get(second_name)
            if first is None or second is None:
                return
            if abs(first.center[1] - second.center[1]) > maximum_vertical_offset:
                return
            left, right = (first, second) if first.center[0] < second.center[0] else (second, first)
            pairs.append(
                SignPair(
                    pair_type=pair_type,
                    left_label=left.class_name,
                    right_label=right.class_name,
                    left_center=left.center,
                    right_center=right.center,
                )
            )

        append_pair("no-entry", "gate", "no-entry_gate")
        append_pair("ben-thanh-station", "suoi-tien-station", "ben-thanh_suoi-tien")
        append_pair("ben-thanh-station", "no-service", "ben-thanh_no-service")
        return pairs

    def infer(self, frame_bgr: np.ndarray) -> SignDetectionResult:
        """Run sign detection and derive semantic flags."""

        result = self._model.predict(
            frame_bgr,
            imgsz=self._settings.image_size,
            device=self._device,
            conf=self._settings.confidence,
            iou=self._settings.iou_threshold,
            verbose=False,
        )[0]
        platforms = self._group_by_platform(result.boxes, frame_bgr.shape)
        has_platform_sign = (
            any(
                int(box.cls[0]) == self._platform_class_id
                and float(box.conf[0]) >= self._settings.confidence
                for box in (list(result.boxes) if result.boxes is not None else [])
            )
            if self._platform_class_id is not None
            else False
        )
        classes = {
            component.class_name for platform in platforms for component in platform.components
        }
        return SignDetectionResult(
            platforms=tuple(platforms),
            has_platform_sign=has_platform_sign,
            has_ticket_sign="ticket-sign" in classes,
            has_gate="gate" in classes,
            has_ben_thanh="ben-thanh-station" in classes,
        )
