"""Shared, dependency-light data structures for the navigation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Protocol

import numpy as np

Point = tuple[int, int]
BoundingBox = tuple[int, int, int, int]


class NavigationPhase(IntEnum):
    """High-level route phase used by the fusion state machine."""

    ENTRY = 1
    AFTER_FIRST_ESCALATOR = 2
    AFTER_TICKET_GATE = 3
    FINAL_PLATFORM = 4


class SegmentationMode(str, Enum):
    """Segmentation guidance mode."""

    SEARCHING = "SEARCHING"
    FOLLOWING = "FOLLOWING"


@dataclass(slots=True, frozen=True)
class ObjectDetection:
    """One object detector result in image coordinates."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    center: Point
    clock_direction: str | None
    angle_degrees: float


@dataclass(slots=True, frozen=True)
class ObjectDetectionResult:
    """All object detections plus the escalator subset used for priority handling."""

    detections: tuple[ObjectDetection, ...] = ()
    escalator_detections: tuple[ObjectDetection, ...] = ()


@dataclass(slots=True, frozen=True)
class SignComponent:
    """One sign-related class detected inside or near a platform-sign region."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    center: tuple[float, float]


@dataclass(slots=True, frozen=True)
class SignPair:
    """A left/right semantic pair inferred from sign detections."""

    pair_type: str
    left_label: str
    right_label: str
    left_center: tuple[float, float]
    right_center: tuple[float, float]


@dataclass(slots=True)
class SignPlatform:
    """A platform-sign region and the components assigned to it."""

    bbox: BoundingBox
    components: list[SignComponent] = field(default_factory=list)
    pairs: list[SignPair] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SignDetectionResult:
    """Sign detector output used by the fusion layer."""

    platforms: tuple[SignPlatform, ...] = ()
    has_platform_sign: bool = False
    has_ticket_sign: bool = False
    has_gate: bool = False
    has_ben_thanh: bool = False


@dataclass(slots=True)
class SegmentationResult:
    """Semantic-segmentation output and candidate guidance targets."""

    mask_bgr: np.ndarray | None
    line_start: Point
    line_end: Point
    root_point: Point
    line_thickness: int
    mode: SegmentationMode
    safe_target: Point | None = None
    lookahead_target: Point | None = None
    standalone_curb_target: Point | None = None
    right_edge_curb_target: Point | None = None
    left_turn_curb_target: Point | None = None
    right_edge_curb_close: bool = False


@dataclass(slots=True, frozen=True)
class GuidanceDecision:
    """Guidance decision returned for one frame."""

    target_point: Point | None = None
    ui_text: str = ""
    clock_direction: str | None = None
    angle_degrees: float = 0.0


class SpeakerProtocol(Protocol):
    """Minimal interface required by the fusion state machine."""

    def say(self, text: str, *, force: bool = False) -> bool:
        """Queue text for speech and return whether it was accepted."""

    def stop(self) -> None:
        """Stop the speaker and release its resources."""
