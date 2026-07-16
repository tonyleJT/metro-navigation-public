"""Runtime video pipeline for the metro navigation prototype."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from metro_navigation.config import Settings
from metro_navigation.core.fusion import NavigationFusion
from metro_navigation.domain import (
    ObjectDetectionResult,
    SignDetectionResult,
)
from metro_navigation.models.object_detector import ObjectDetector
from metro_navigation.models.segmenter import SegFormerGuidance
from metro_navigation.models.sign_detector import SignDetector
from metro_navigation.utils.speech import NullSpeaker, Speaker
from metro_navigation.visualization import render_overlay

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeTiming:
    frame_index: int
    ran_segmentation: bool
    ran_object_detection: bool
    ran_sign_detection: bool
    phase: int
    segmentation_enabled: bool
    segmentation_ms: float
    object_detection_ms: float
    sign_detection_ms: float
    fusion_ms: float
    visualization_ms: float
    frame_ms: float
    ui_text: str


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _should_run(frame_index: int, interval: int, *, no_cached_value: bool) -> bool:
    return no_cached_value or interval <= 1 or frame_index % interval == 0


def _parse_capture_source(source: str) -> str | int:
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def _open_video_writer(
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open output video writer: {output_path}")
    return writer


def _write_runtime_log(path: Path, rows: list[RuntimeTiming]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run_video_pipeline(
    *,
    source: str,
    settings: Settings,
    display: bool,
    output_video: Path | None,
    sample_interval: int,
    sample_directory: Path,
    runtime_log: Path,
    speech_enabled: bool,
) -> dict[str, Any]:
    """Run the complete inference pipeline and return summary statistics."""

    settings.models.validate()
    capture = cv2.VideoCapture(_parse_capture_source(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video/camera source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The input source reported an invalid frame size")

    writer = (
        _open_video_writer(
            output_video,
            fps=source_fps,
            width=width,
            height=height,
        )
        if output_video is not None
        else None
    )
    if sample_interval > 0:
        sample_directory.mkdir(parents=True, exist_ok=True)

    speaker = Speaker(settings.speech) if speech_enabled else NullSpeaker()
    object_detector: ObjectDetector | None = None
    segmenter: SegFormerGuidance | None = None
    sign_detector: SignDetector | None = None
    timing_rows: list[RuntimeTiming] = []
    processed_frames = 0
    frame_index = 0
    total_start = time.perf_counter()

    try:
        LOGGER.info("Loading model weights")
        object_detector = ObjectDetector(
            settings.models.object_detector,
            device=settings.device,
            settings=settings.object_detector,
        )
        segmenter = SegFormerGuidance(
            settings.models.segmenter,
            backbone=settings.models.segmenter_backbone,
            device=settings.device,
            settings=settings.segmentation,
        )
        sign_detector = SignDetector(
            settings.models.sign_detector,
            device=settings.device,
            settings=settings.sign_detector,
        )
        fusion = NavigationFusion(speaker, settings.fusion)

        last_object_result: ObjectDetectionResult | None = None
        last_segmentation_result = None
        last_sign_result = SignDetectionResult()
        last_segmentation_enabled: bool | None = None
        last_segmentation_phase = None
        visual_mode = display or writer is not None or sample_interval > 0

        LOGGER.info("Starting inference")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_start = time.perf_counter()

            run_segmentation = _should_run(
                frame_index,
                settings.runtime.segmentation_interval,
                no_cached_value=last_segmentation_result is None,
            )
            run_segmentation = run_segmentation or (
                last_segmentation_enabled != fusion.segmentation_enabled
                or last_segmentation_phase != fusion.phase
            )
            segmentation_ms = 0.0
            if run_segmentation:
                synchronize_cuda()
                start = time.perf_counter()
                last_segmentation_result = segmenter.update(
                    frame,
                    enabled=fusion.segmentation_enabled,
                    phase=fusion.phase,
                    make_mask=visual_mode,
                )
                synchronize_cuda()
                segmentation_ms = (time.perf_counter() - start) * 1_000.0
                last_segmentation_enabled = fusion.segmentation_enabled
                last_segmentation_phase = fusion.phase
            segmentation_result = last_segmentation_result
            assert segmentation_result is not None

            run_object_detection = _should_run(
                frame_index,
                settings.runtime.object_detection_interval,
                no_cached_value=last_object_result is None,
            )
            object_detection_ms = 0.0
            if run_object_detection:
                synchronize_cuda()
                start = time.perf_counter()
                last_object_result = object_detector.infer(
                    frame,
                    root_point=segmentation_result.root_point,
                )
                synchronize_cuda()
                object_detection_ms = (time.perf_counter() - start) * 1_000.0
            object_result = last_object_result
            assert object_result is not None

            run_sign_detection = _should_run(
                frame_index,
                settings.runtime.sign_detection_interval,
                no_cached_value=frame_index == 0,
            )
            sign_detection_ms = 0.0
            if run_sign_detection:
                synchronize_cuda()
                start = time.perf_counter()
                last_sign_result = sign_detector.infer(frame)
                synchronize_cuda()
                sign_detection_ms = (time.perf_counter() - start) * 1_000.0

            start = time.perf_counter()
            decision = fusion.update(
                object_result,
                segmentation_result,
                last_sign_result,
            )
            fusion_ms = (time.perf_counter() - start) * 1_000.0

            visualization_ms = 0.0
            if visual_mode:
                start = time.perf_counter()
                overlay = render_overlay(
                    frame,
                    segmentation_result,
                    object_result,
                    last_sign_result,
                    decision,
                    arrow_length_pixels=settings.runtime.arrow_length_pixels,
                )
                if writer is not None:
                    writer.write(overlay)
                if sample_interval > 0 and frame_index % sample_interval == 0:
                    sample_path = sample_directory / f"frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(sample_path), overlay):
                        LOGGER.warning("Failed to save sample frame: %s", sample_path)
                if display:
                    cv2.imshow("Metro Navigation", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                visualization_ms = (time.perf_counter() - start) * 1_000.0

            frame_ms = (time.perf_counter() - frame_start) * 1_000.0
            timing_rows.append(
                RuntimeTiming(
                    frame_index=frame_index,
                    ran_segmentation=run_segmentation,
                    ran_object_detection=run_object_detection,
                    ran_sign_detection=run_sign_detection,
                    phase=int(fusion.phase),
                    segmentation_enabled=fusion.segmentation_enabled,
                    segmentation_ms=segmentation_ms,
                    object_detection_ms=object_detection_ms,
                    sign_detection_ms=sign_detection_ms,
                    fusion_ms=fusion_ms,
                    visualization_ms=visualization_ms,
                    frame_ms=frame_ms,
                    ui_text=decision.ui_text,
                )
            )
            processed_frames += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        speaker.stop()

    total_seconds = time.perf_counter() - total_start
    average_fps = processed_frames / max(total_seconds, 1e-9)
    _write_runtime_log(runtime_log, timing_rows)
    return {
        "frames": processed_frames,
        "total_seconds": total_seconds,
        "average_fps": average_fps,
        "average_ms_per_frame": 1_000.0 / max(average_fps, 1e-9),
        "runtime_log": str(runtime_log),
        "output_video": str(output_video) if output_video is not None else None,
    }
