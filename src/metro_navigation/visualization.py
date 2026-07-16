"""OpenCV visualization helpers for debugging and demonstration videos."""

from __future__ import annotations

import cv2
import numpy as np

from metro_navigation.domain import (
    GuidanceDecision,
    ObjectDetectionResult,
    Point,
    SegmentationResult,
    SignDetectionResult,
)

OBJECT_CLASSES_TO_DRAW = frozenset({"stair node", "escalator entry node", "ticket booth"})


def _draw_arrow(
    image: np.ndarray,
    root_point: Point,
    target_point: Point,
    *,
    length_pixels: int,
) -> None:
    root_x, root_y = root_point
    target_x, target_y = target_point
    delta_x = target_x - root_x
    delta_y = target_y - root_y
    distance = max(float(np.hypot(delta_x, delta_y)), 1e-6)
    scale = length_pixels / distance
    end_point = (
        int(root_x + delta_x * scale),
        int(root_y + delta_y * scale),
    )
    cv2.arrowedLine(
        image,
        root_point,
        end_point,
        (255, 255, 255),
        2,
        tipLength=0.2,
    )


def _draw_announcement(image: np.ndarray, text: str) -> None:
    if not text:
        return
    origin = (20, 45)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_object_detections(
    image: np.ndarray,
    result: ObjectDetectionResult,
) -> None:
    for detection in result.detections:
        if detection.class_name not in OBJECT_CLASSES_TO_DRAW:
            continue
        x1, y1, x2, y2 = detection.bbox
        color = (0, 0, 255) if detection.class_name == "escalator entry node" else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            detection.class_name,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_sign_detections(
    image: np.ndarray,
    result: SignDetectionResult,
) -> None:
    for platform in result.platforms:
        x1, y1, x2, y2 = platform.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if result.has_platform_sign:
            cv2.putText(
                image,
                "platform sign",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for component in platform.components:
            component_x1, component_y1, component_x2, component_y2 = component.bbox
            cv2.rectangle(
                image,
                (component_x1, component_y1),
                (component_x2, component_y2),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                image,
                f"{component.class_name} {component.confidence:.2f}",
                (component_x1, max(0, component_y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for pair in platform.pairs:
            left = tuple(int(value) for value in pair.left_center)
            right = tuple(int(value) for value in pair.right_center)
            cv2.circle(image, left, 4, (255, 255, 255), -1)
            cv2.circle(image, right, 4, (255, 255, 255), -1)
            cv2.putText(
                image,
                "L",
                (left[0] - 6, left[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "R",
                (right[0] - 6, right[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def render_overlay(
    frame_bgr: np.ndarray,
    segmentation_result: SegmentationResult,
    object_result: ObjectDetectionResult,
    sign_result: SignDetectionResult,
    decision: GuidanceDecision,
    *,
    arrow_length_pixels: int,
) -> np.ndarray:
    """Render the current frame with perception and guidance overlays."""

    if segmentation_result.mask_bgr is not None:
        image = cv2.addWeighted(
            frame_bgr,
            0.6,
            segmentation_result.mask_bgr,
            0.4,
            0.0,
        )
    else:
        image = frame_bgr.copy()

    _draw_object_detections(image, object_result)
    _draw_sign_detections(image, sign_result)
    cv2.line(
        image,
        segmentation_result.line_start,
        segmentation_result.line_end,
        (0, 255, 0),
        segmentation_result.line_thickness,
    )
    cv2.circle(image, segmentation_result.root_point, 6, (0, 255, 0), -1)

    if decision.target_point is not None:
        cv2.circle(image, decision.target_point, 6, (255, 255, 255), -1)
        _draw_arrow(
            image,
            segmentation_result.root_point,
            decision.target_point,
            length_pixels=arrow_length_pixels,
        )

    _draw_announcement(image, decision.ui_text)
    return image
