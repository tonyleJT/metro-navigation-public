"""Typed configuration for the metro navigation research prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = REPOSITORY_ROOT / "models" / "weights"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def default_device() -> str:
    """Return the default inference device available on this machine."""

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True, frozen=True)
class ModelPaths:
    """Paths to custom model weights."""

    object_detector: Path = field(
        default_factory=lambda: _env_path(
            "METRO_NAV_OD_WEIGHTS", DEFAULT_WEIGHTS_DIR / "yolo11m.pt"
        )
    )
    segmenter: Path = field(
        default_factory=lambda: _env_path(
            "METRO_NAV_SEGMENTER_WEIGHTS", DEFAULT_WEIGHTS_DIR / "segformer.pt"
        )
    )
    sign_detector: Path = field(
        default_factory=lambda: _env_path(
            "METRO_NAV_SIGN_WEIGHTS", DEFAULT_WEIGHTS_DIR / "yoloOCR.pt"
        )
    )
    segmenter_backbone: str = os.getenv(
        "METRO_NAV_SEGMENTER_BACKBONE",
        "nvidia/segformer-b0-finetuned-ade-512-512",
    )

    def with_overrides(
        self,
        *,
        object_detector: Path | None = None,
        segmenter: Path | None = None,
        sign_detector: Path | None = None,
    ) -> ModelPaths:
        """Return a copy with selected paths replaced."""

        return replace(
            self,
            object_detector=object_detector or self.object_detector,
            segmenter=segmenter or self.segmenter,
            sign_detector=sign_detector or self.sign_detector,
        )

    def validate(self) -> None:
        """Raise a clear error when required custom weights are missing."""

        missing = [
            path
            for path in (
                self.object_detector,
                self.segmenter,
                self.sign_detector,
            )
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Required model weights were not found:\n"
                f"{formatted}\n"
                "Place the files under models/weights or override the paths with "
                "CLI options/environment variables."
            )


@dataclass(slots=True, frozen=True)
class ObjectDetectorSettings:
    image_size: int = 640
    default_confidence: float = 0.60
    high_confidence: float = 0.80
    high_confidence_classes: frozenset[str] = frozenset(
        {"ticket booth", "stair node", "pillar node", "escalator entry node"}
    )


@dataclass(slots=True, frozen=True)
class SignDetectorSettings:
    image_size: int = 640
    confidence: float = 0.50
    iou_threshold: float = 0.45
    pair_max_vertical_offset_ratio: float = 0.25


@dataclass(slots=True, frozen=True)
class SegmentationSettings:
    image_size: int = 512
    morphology_kernel_size: int = 5
    minimum_safe_area_pixels: int = 2_000
    forward_weight: float = 1.0
    lateral_weight: float = 0.5
    lookahead_max_ahead_ratio: float = 0.50
    lookahead_min_ahead_ratio: float = 0.15
    lookahead_roi_half_width_ratio: float = 0.25
    lookahead_minimum_pixels: int = 30
    lookahead_target_ratio: float = 0.38
    lookahead_window_ratio: float = 0.10
    lookahead_ema_alpha: float = 0.73
    mode_on_frames: int = 2
    mode_off_frames: int = 4
    right_edge_close_distance_ratio: float = 0.12


@dataclass(slots=True, frozen=True)
class RuntimeSettings:
    object_detection_interval: int = 1
    segmentation_interval: int = 1
    sign_detection_interval: int = 8
    arrow_length_pixels: int = 140


@dataclass(slots=True, frozen=True)
class FusionSettings:
    changed_guidance_cooldown_seconds: float = 4.0
    repeated_guidance_interval_seconds: float = 8.0
    phase_confirmation_frames: int = 3
    stage_two_turn_delay_seconds: float = 2.0
    ticket_booth_pass_seconds: float = 3.0
    destination_repeat_seconds: float = 25.0
    stage_two_straight_angle_degrees: float = 15.0
    station_name: str | None = None


@dataclass(slots=True, frozen=True)
class SpeechSettings:
    global_cooldown_seconds: float = 1.0
    rate_words_per_minute: int = 170
    volume: float = 1.0


@dataclass(slots=True, frozen=True)
class Settings:
    """Complete application configuration."""

    device: str = field(default_factory=default_device)
    models: ModelPaths = field(default_factory=ModelPaths)
    object_detector: ObjectDetectorSettings = field(default_factory=ObjectDetectorSettings)
    sign_detector: SignDetectorSettings = field(default_factory=SignDetectorSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
