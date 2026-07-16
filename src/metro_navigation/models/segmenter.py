"""SegFormer-B0 semantic segmentation and path-target extraction."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from metro_navigation.config import SegmentationSettings
from metro_navigation.domain import (
    NavigationPhase,
    Point,
    SegmentationMode,
    SegmentationResult,
)
from metro_navigation.utils.clock import anchor_line_geometry

LOGGER = logging.getLogger(__name__)
ID_TO_LABEL = {0: "background", 1: "blindway", 2: "curb_ramp"}
LABEL_TO_ID = {label: class_id for class_id, label in ID_TO_LABEL.items()}
COLOR_LOOKUP = np.array(
    [
        (128, 128, 128),
        (0, 255, 255),
        (0, 0, 255),
    ],
    dtype=np.uint8,
)


@dataclass(slots=True, frozen=True)
class _SafeRegion:
    center: Point
    area: float


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


class SegFormerGuidance:
    """Run semantic segmentation and derive SEARCHING/FOLLOWING targets."""

    ANCHOR_LINE_LENGTH_RATIO = 0.23
    ANCHOR_LINE_THICKNESS_PIXELS = 10
    ANCHOR_LINE_VERTICAL_OFFSET_PIXELS = 6

    STANDALONE_CURB_MINIMUM_AREA = 800.0
    LEFT_TURN_CURB_MINIMUM_AREA = 200.0
    RIGHT_EDGE_CURB_MINIMUM_AREA = 150.0

    def __init__(
        self,
        weights_path: Path,
        *,
        backbone: str,
        device: str,
        settings: SegmentationSettings,
    ) -> None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Segmentation weights not found: {weights_path}")

        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self._device = torch.device(device)
        self._settings = settings
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=len(ID_TO_LABEL),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )
        processor = SegformerImageProcessor.from_pretrained(backbone)
        processor.do_reduce_labels = False

        checkpoint = torch.load(
            weights_path,
            map_location=self._device,
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Segmentation checkpoint must contain a state-dict mapping")
        state_dict_raw = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict_raw, Mapping):
            raise TypeError("model_state_dict must be a mapping")
        state_dict = {
            str(key).removeprefix("model."): value for key, value in state_dict_raw.items()
        }
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device)
        self._model.eval()
        self._model = self._model.to(memory_format=torch.channels_last)

        self._mean = torch.tensor(
            processor.image_mean,
            device=self._device,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            processor.image_std,
            device=self._device,
        ).view(1, 3, 1, 1)

        self._use_cuda = self._device.type == "cuda" and torch.cuda.is_available()
        if self._use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._mode = SegmentationMode.SEARCHING
        self._on_counter = 0
        self._off_counter = 0
        self._last_lookahead_x: int | None = None

        close_size = _odd_at_least(settings.morphology_kernel_size)
        open_size = _odd_at_least(settings.morphology_kernel_size - 2)
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )
        self._curb_kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._curb_kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @property
    def mode(self) -> SegmentationMode:
        return self._mode

    def reset_tracking(self) -> None:
        """Reset temporal SEARCHING/FOLLOWING state between independent runs."""

        self._mode = SegmentationMode.SEARCHING
        self._on_counter = 0
        self._off_counter = 0
        self._last_lookahead_x = None

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            frame_rgb,
            (self._settings.image_size, self._settings.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(resized).to(self._device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        tensor = tensor.to(memory_format=torch.channels_last, dtype=torch.float32)
        tensor = tensor / 255.0
        return (tensor - self._mean) / self._std

    def _segment(
        self,
        pixel_values: torch.Tensor,
        *,
        output_height: int,
        output_width: int,
        make_mask: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), autocast:
            logits = self._model(pixel_values).logits
            resized_logits = functional.interpolate(
                logits,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = resized_logits.argmax(dim=1).to(torch.int16)

        class_map = prediction.squeeze(0).cpu().numpy()
        mask_bgr = COLOR_LOOKUP[class_map] if make_mask else None
        return mask_bgr, class_map

    def _anchor_contacts(
        self,
        class_map: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> tuple[Point, Point, Point, int, bool, bool]:
        line_start, line_end, root_point, thickness = anchor_line_geometry(
            width,
            height,
            length_ratio=self.ANCHOR_LINE_LENGTH_RATIO,
            thickness_pixels=self.ANCHOR_LINE_THICKNESS_PIXELS,
            vertical_offset_pixels=self.ANCHOR_LINE_VERTICAL_OFFSET_PIXELS,
        )
        x1, line_y = line_start
        x2, _ = line_end
        half_thickness = max(1, thickness // 2)
        y0 = max(0, line_y - half_thickness)
        y1 = min(height, line_y + half_thickness + 1)
        line_region = class_map[y0:y1, x1 : x2 + 1]
        return (
            line_start,
            line_end,
            root_point,
            thickness,
            bool((line_region == 1).any()),
            bool((line_region == 2).any()),
        )

    def _update_mode(self, touches_blindway: bool) -> None:
        if self._mode is SegmentationMode.SEARCHING:
            self._on_counter = self._on_counter + 1 if touches_blindway else 0
            self._off_counter = 0
            if self._on_counter >= self._settings.mode_on_frames:
                self._mode = SegmentationMode.FOLLOWING
                self._on_counter = 0
                self._last_lookahead_x = None
            return

        self._off_counter = self._off_counter + 1 if not touches_blindway else 0
        self._on_counter = 0
        if self._off_counter >= self._settings.mode_off_frames:
            self._mode = SegmentationMode.SEARCHING
            self._off_counter = 0
            self._last_lookahead_x = None

    def _find_safe_regions(self, class_map: np.ndarray) -> list[_SafeRegion]:
        safe_mask = np.where((class_map == 1) | (class_map == 2), 255, 0).astype(np.uint8)
        safe_mask = cv2.morphologyEx(
            safe_mask,
            cv2.MORPH_CLOSE,
            self._close_kernel,
            iterations=1,
        )
        safe_mask = cv2.morphologyEx(
            safe_mask,
            cv2.MORPH_OPEN,
            self._open_kernel,
            iterations=1,
        )
        contours, _ = cv2.findContours(
            safe_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: list[_SafeRegion] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._settings.minimum_safe_area_pixels:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            regions.append(
                _SafeRegion(
                    center=(
                        int(moments["m10"] / moments["m00"]),
                        int(moments["m01"] / moments["m00"]),
                    ),
                    area=area,
                )
            )
        return regions

    def _choose_best_safe_region(
        self,
        regions: list[_SafeRegion],
        *,
        width: int,
        height: int,
        root_point: Point,
    ) -> Point | None:
        root_x, root_y = root_point
        best_target: Point | None = None
        best_score = float("-inf")

        for region in regions:
            center_x, center_y = region.center
            lateral_offset = center_x - root_x
            forward_distance = root_y - center_y
            if forward_distance <= 0:
                continue

            forward_closeness = float(np.clip(1.0 - forward_distance / float(height), 0.0, 1.0))
            lateral_penalty = min(abs(lateral_offset) / (width / 2.0), 1.0)
            score = (
                self._settings.forward_weight * forward_closeness
                - self._settings.lateral_weight * lateral_penalty
            )
            if score > best_score:
                best_score = score
                best_target = region.center
        return best_target

    def _find_lookahead_target(
        self,
        class_map: np.ndarray,
        *,
        width: int,
        height: int,
        root_point: Point,
    ) -> Point | None:
        root_x, root_y = root_point
        y_min = max(
            0,
            int(root_y - height * self._settings.lookahead_max_ahead_ratio),
        )
        y_max = max(
            0,
            int(root_y - height * self._settings.lookahead_min_ahead_ratio),
        )
        if y_max <= y_min:
            return None

        x_min = max(
            0,
            int(root_x - width * self._settings.lookahead_roi_half_width_ratio),
        )
        x_max = min(
            width - 1,
            int(root_x + width * self._settings.lookahead_roi_half_width_ratio),
        )
        target_y = int(root_y - height * self._settings.lookahead_target_ratio)
        target_y = int(np.clip(target_y, y_min, y_max))
        window = int(height * self._settings.lookahead_window_ratio)
        search_y0 = max(y_min, target_y - window)
        search_y1 = min(y_max, target_y + window)

        best_y: int | None = None
        best_count = -1
        best_mean_x: int | None = None
        best_distance = float("inf")

        for row_y in range(search_y0, search_y1 + 1):
            row = class_map[row_y, x_min : x_max + 1]
            blindway_pixels = row == 1
            count = int(blindway_pixels.sum())
            if count < self._settings.lookahead_minimum_pixels:
                continue
            distance = abs(row_y - target_y)
            if count > best_count or (count == best_count and distance < best_distance):
                x_values = np.where(blindway_pixels)[0] + x_min
                best_mean_x = int(x_values.mean())
                best_y = row_y
                best_count = count
                best_distance = distance

        if best_y is None or best_mean_x is None:
            return None

        if self._last_lookahead_x is None:
            smoothed_x = best_mean_x
        else:
            alpha = self._settings.lookahead_ema_alpha
            smoothed_x = int(alpha * self._last_lookahead_x + (1.0 - alpha) * best_mean_x)
        self._last_lookahead_x = smoothed_x
        return smoothed_x, best_y

    def _largest_curb_centroid(
        self,
        class_map: np.ndarray,
        *,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        kernel: np.ndarray,
        minimum_area: float,
    ) -> Point | None:
        if x_max <= x_min or y_max <= y_min:
            return None
        region = class_map[y_min:y_max, x_min:x_max]
        curb_mask = np.where(region == 2, 255, 0).astype(np.uint8)
        if not curb_mask.any():
            return None
        curb_mask = cv2.morphologyEx(
            curb_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        contours, _ = cv2.findContours(
            curb_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < minimum_area:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        return (
            x_min + int(moments["m10"] / moments["m00"]),
            y_min + int(moments["m01"] / moments["m00"]),
        )

    def _find_standalone_curb_target(
        self,
        class_map: np.ndarray,
        *,
        width: int,
        height: int,
        root_point: Point,
    ) -> Point | None:
        root_x, root_y = root_point
        return self._largest_curb_centroid(
            class_map,
            x_min=max(0, int(root_x - width * 0.50)),
            x_max=max(0, root_x - 1),
            y_min=max(0, int(root_y - height * 0.50)),
            y_max=max(0, int(root_y - height * 0.15)),
            kernel=self._curb_kernel_5,
            minimum_area=self.STANDALONE_CURB_MINIMUM_AREA,
        )

    def _find_left_turn_curb_target(
        self,
        class_map: np.ndarray,
        *,
        width: int,
        height: int,
        root_point: Point,
    ) -> Point | None:
        root_x, root_y = root_point
        return self._largest_curb_centroid(
            class_map,
            x_min=max(0, int(root_x - width * 0.50)),
            x_max=max(0, root_x - 1),
            y_min=max(0, int(root_y - height * 0.25)),
            y_max=min(height, int(root_y + height * 0.05)),
            kernel=self._curb_kernel_7,
            minimum_area=self.LEFT_TURN_CURB_MINIMUM_AREA,
        )

    def _find_right_edge_curb_target(
        self,
        class_map: np.ndarray,
        *,
        width: int,
        height: int,
        root_point: Point,
    ) -> Point | None:
        _, root_y = root_point
        y_min = max(0, int(root_y - height * 0.60))
        y_max = max(0, int(root_y - height * 0.05))
        x_min = int(width * 0.55)
        x_max = width
        if x_max <= x_min or y_max <= y_min:
            return None

        region = class_map[y_min:y_max, x_min:x_max]
        curb_mask = np.where(region == 2, 255, 0).astype(np.uint8)
        if not curb_mask.any():
            return None
        curb_mask = cv2.morphologyEx(
            curb_mask,
            cv2.MORPH_OPEN,
            self._curb_kernel_5,
            iterations=1,
        )
        contours, _ = cv2.findContours(
            curb_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[Point] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.RIGHT_EDGE_CURB_MINIMUM_AREA:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            candidates.append(
                (
                    x_min + int(moments["m10"] / moments["m00"]),
                    y_min + int(moments["m01"] / moments["m00"]),
                )
            )
        return max(candidates, key=lambda point: point[0], default=None)

    def update(
        self,
        frame_bgr: np.ndarray,
        *,
        enabled: bool,
        phase: NavigationPhase,
        make_mask: bool = True,
    ) -> SegmentationResult:
        """Process one frame and return guidance candidates."""

        height, width = frame_bgr.shape[:2]
        line_start, line_end, root_point, thickness = anchor_line_geometry(
            width,
            height,
            length_ratio=self.ANCHOR_LINE_LENGTH_RATIO,
            thickness_pixels=self.ANCHOR_LINE_THICKNESS_PIXELS,
            vertical_offset_pixels=self.ANCHOR_LINE_VERTICAL_OFFSET_PIXELS,
        )
        if not enabled:
            return SegmentationResult(
                mask_bgr=None,
                line_start=line_start,
                line_end=line_end,
                root_point=root_point,
                line_thickness=thickness,
                mode=self._mode,
            )

        pixel_values = self._preprocess(frame_bgr)
        mask_bgr, class_map = self._segment(
            pixel_values,
            output_height=height,
            output_width=width,
            make_mask=make_mask,
        )
        (
            line_start,
            line_end,
            root_point,
            thickness,
            touches_blindway,
            touches_curb,
        ) = self._anchor_contacts(class_map, width=width, height=height)
        self._update_mode(touches_blindway)

        safe_target: Point | None = None
        lookahead_target: Point | None = None
        if self._mode is SegmentationMode.SEARCHING:
            safe_target = self._choose_best_safe_region(
                self._find_safe_regions(class_map),
                width=width,
                height=height,
                root_point=root_point,
            )
        else:
            lookahead_target = self._find_lookahead_target(
                class_map,
                width=width,
                height=height,
                root_point=root_point,
            )

        standalone_curb_target = None
        if phase in {
            NavigationPhase.ENTRY,
            NavigationPhase.AFTER_TICKET_GATE,
        }:
            standalone_curb_target = self._find_standalone_curb_target(
                class_map,
                width=width,
                height=height,
                root_point=root_point,
            )

        right_edge_curb_target = self._find_right_edge_curb_target(
            class_map,
            width=width,
            height=height,
            root_point=root_point,
        )
        left_turn_curb_target = None
        if phase is NavigationPhase.FINAL_PLATFORM:
            left_turn_curb_target = self._find_left_turn_curb_target(
                class_map,
                width=width,
                height=height,
                root_point=root_point,
            )

        right_edge_curb_close = False
        if right_edge_curb_target is not None:
            delta_x = right_edge_curb_target[0] - root_point[0]
            delta_y = right_edge_curb_target[1] - root_point[1]
            distance = float(np.hypot(delta_x, delta_y))
            right_edge_curb_close = (
                distance < self._settings.right_edge_close_distance_ratio * height or touches_curb
            )

        return SegmentationResult(
            mask_bgr=mask_bgr,
            line_start=line_start,
            line_end=line_end,
            root_point=root_point,
            line_thickness=thickness,
            mode=self._mode,
            safe_target=safe_target,
            lookahead_target=lookahead_target,
            standalone_curb_target=standalone_curb_target,
            right_edge_curb_target=right_edge_curb_target,
            left_turn_curb_target=left_turn_curb_target,
            right_edge_curb_close=right_edge_curb_close,
        )
