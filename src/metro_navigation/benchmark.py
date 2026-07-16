"""Streaming component benchmark that avoids loading the full video into RAM."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import ParamSpec, TypeVar

import cv2
import numpy as np

from metro_navigation.config import Settings
from metro_navigation.core.fusion import NavigationFusion
from metro_navigation.domain import SignDetectionResult
from metro_navigation.models.object_detector import ObjectDetector
from metro_navigation.models.segmenter import SegFormerGuidance
from metro_navigation.models.sign_detector import SignDetector
from metro_navigation.pipeline import synchronize_cuda
from metro_navigation.utils.clock import anchor_line_geometry
from metro_navigation.utils.speech import NullSpeaker

LOGGER = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    component: str
    input_size: str
    mean_latency_ms: float
    standard_deviation_ms: float
    fps_equivalent: float
    measured_frames: int


@dataclass(slots=True, frozen=True)
class RawTiming:
    component: str
    frame_index: int
    latency_ms: float


def _video_frames(source: str, max_frames: int) -> Iterator[np.ndarray]:
    if source.strip().isdigit():
        raise ValueError("Benchmarking requires a video file, not a live camera index")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open benchmark video: {source}")
    count = 0
    try:
        while max_frames <= 0 or count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
            count += 1
    finally:
        capture.release()


def _measure_call(
    call: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[T, float]:
    synchronize_cuda()
    start = time.perf_counter()
    result = call(*args, **kwargs)
    synchronize_cuda()
    return result, (time.perf_counter() - start) * 1_000.0


def _summarize(component: str, input_size: str, times: list[float]) -> BenchmarkSummary:
    average = mean(times) if times else 0.0
    deviation = stdev(times) if len(times) > 1 else 0.0
    return BenchmarkSummary(
        component=component,
        input_size=input_size,
        mean_latency_ms=average,
        standard_deviation_ms=deviation,
        fps_equivalent=1_000.0 / average if average > 0 else 0.0,
        measured_frames=len(times),
    )


def _append_raw(rows: list[RawTiming], component: str, times: list[float]) -> None:
    rows.extend(
        RawTiming(component=component, frame_index=index, latency_ms=latency)
        for index, latency in enumerate(times)
    )


def _benchmark_object_detector(
    source: str,
    *,
    max_frames: int,
    warmup_frames: int,
    detector: ObjectDetector,
) -> list[float]:
    timings: list[float] = []
    for frame_index, frame in enumerate(_video_frames(source, max_frames)):
        height, width = frame.shape[:2]
        _, _, root_point, _ = anchor_line_geometry(width, height)
        _, latency = _measure_call(detector.infer, frame, root_point=root_point)
        if frame_index >= warmup_frames:
            timings.append(latency)
    return timings


def _benchmark_sign_detector(
    source: str,
    *,
    max_frames: int,
    warmup_frames: int,
    detector: SignDetector,
) -> list[float]:
    timings: list[float] = []
    for frame_index, frame in enumerate(_video_frames(source, max_frames)):
        _, latency = _measure_call(detector.infer, frame)
        if frame_index >= warmup_frames:
            timings.append(latency)
    return timings


def _benchmark_segmenter(
    source: str,
    *,
    max_frames: int,
    warmup_frames: int,
    segmenter: SegFormerGuidance,
) -> list[float]:
    segmenter.reset_tracking()
    timings: list[float] = []
    from metro_navigation.domain import NavigationPhase

    for frame_index, frame in enumerate(_video_frames(source, max_frames)):
        _, latency = _measure_call(
            segmenter.update,
            frame,
            enabled=True,
            phase=NavigationPhase.ENTRY,
            make_mask=False,
        )
        if frame_index >= warmup_frames:
            timings.append(latency)
    return timings


def _benchmark_fusion(
    source: str,
    *,
    max_frames: int,
    warmup_frames: int,
    object_detector: ObjectDetector,
    segmenter: SegFormerGuidance,
    sign_detector: SignDetector,
    settings: Settings,
) -> list[float]:
    segmenter.reset_tracking()
    cache_fusion = NavigationFusion(NullSpeaker(), settings.fusion)
    cached_inputs = []
    for frame in _video_frames(source, max_frames):
        segmentation_result = segmenter.update(
            frame,
            enabled=cache_fusion.segmentation_enabled,
            phase=cache_fusion.phase,
            make_mask=False,
        )
        object_result = object_detector.infer(
            frame,
            root_point=segmentation_result.root_point,
        )
        sign_result = sign_detector.infer(frame)
        cached_inputs.append((object_result, segmentation_result, sign_result))
        cache_fusion.update(object_result, segmentation_result, sign_result)

    fusion = NavigationFusion(NullSpeaker(), settings.fusion)
    timings: list[float] = []
    for frame_index, inputs in enumerate(cached_inputs):
        _, latency = _measure_call(fusion.update, *inputs)
        if frame_index >= warmup_frames:
            timings.append(latency)
    return timings


def _benchmark_full_pipeline(
    source: str,
    *,
    max_frames: int,
    warmup_frames: int,
    sign_interval: int,
    object_detector: ObjectDetector,
    segmenter: SegFormerGuidance,
    sign_detector: SignDetector,
    settings: Settings,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    segmenter.reset_tracking()
    fusion = NavigationFusion(NullSpeaker(), settings.fusion)
    last_sign_result = SignDetectionResult()
    full_times: list[float] = []
    object_times: list[float] = []
    segmentation_times: list[float] = []
    sign_times: list[float] = []
    fusion_times: list[float] = []

    for frame_index, frame in enumerate(_video_frames(source, max_frames)):
        synchronize_cuda()
        frame_start = time.perf_counter()

        segmentation_result, segmentation_latency = _measure_call(
            segmenter.update,
            frame,
            enabled=fusion.segmentation_enabled,
            phase=fusion.phase,
            make_mask=False,
        )
        object_result, object_latency = _measure_call(
            object_detector.infer,
            frame,
            root_point=segmentation_result.root_point,
        )

        sign_latency = 0.0
        if sign_interval <= 1 or frame_index % sign_interval == 0:
            last_sign_result, sign_latency = _measure_call(sign_detector.infer, frame)

        _, fusion_latency = _measure_call(
            fusion.update,
            object_result,
            segmentation_result,
            last_sign_result,
        )
        synchronize_cuda()
        full_latency = (time.perf_counter() - frame_start) * 1_000.0

        if frame_index >= warmup_frames:
            full_times.append(full_latency)
            object_times.append(object_latency)
            segmentation_times.append(segmentation_latency)
            sign_times.append(sign_latency)
            fusion_times.append(fusion_latency)

    return (
        full_times,
        object_times,
        segmentation_times,
        sign_times,
        fusion_times,
    )


def _write_dataclass_csv(
    path: Path,
    rows: list[BenchmarkSummary] | list[RawTiming],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dictionaries[0].keys()))
        writer.writeheader()
        writer.writerows(dictionaries)


def run_benchmark(
    *,
    source: str,
    settings: Settings,
    warmup_frames: int,
    max_frames: int,
    sign_interval: int,
    output_directory: Path,
) -> list[BenchmarkSummary]:
    """Benchmark modules and save summary/raw CSV files."""

    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    settings.models.validate()
    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading models")
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

    summaries: list[BenchmarkSummary] = []
    raw_rows: list[RawTiming] = []

    LOGGER.info("Benchmarking object detector")
    object_times = _benchmark_object_detector(
        source,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        detector=object_detector,
    )
    summaries.append(
        _summarize(
            "YOLO11 object detection",
            f"{settings.object_detector.image_size}x{settings.object_detector.image_size}",
            object_times,
        )
    )
    _append_raw(raw_rows, "YOLO11 object detection", object_times)

    LOGGER.info("Benchmarking sign detector")
    sign_times = _benchmark_sign_detector(
        source,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        detector=sign_detector,
    )
    summaries.append(
        _summarize(
            "YOLO sign detection",
            f"{settings.sign_detector.image_size}x{settings.sign_detector.image_size}",
            sign_times,
        )
    )
    _append_raw(raw_rows, "YOLO sign detection", sign_times)

    LOGGER.info("Benchmarking SegFormer")
    segmentation_times = _benchmark_segmenter(
        source,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        segmenter=segmenter,
    )
    summaries.append(
        _summarize(
            "SegFormer-B0 segmentation",
            f"{settings.segmentation.image_size}x{settings.segmentation.image_size}",
            segmentation_times,
        )
    )
    _append_raw(raw_rows, "SegFormer-B0 segmentation", segmentation_times)

    LOGGER.info("Benchmarking fusion logic")
    fusion_times = _benchmark_fusion(
        source,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        object_detector=object_detector,
        segmenter=segmenter,
        sign_detector=sign_detector,
        settings=settings,
    )
    summaries.append(_summarize("Fusion and guidance logic", "-", fusion_times))
    _append_raw(raw_rows, "Fusion and guidance logic", fusion_times)

    LOGGER.info("Benchmarking full pipeline")
    full_times, _, _, _, _ = _benchmark_full_pipeline(
        source,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        sign_interval=sign_interval,
        object_detector=object_detector,
        segmenter=segmenter,
        sign_detector=sign_detector,
        settings=settings,
    )
    summaries.append(_summarize("Full online pipeline", "mixed", full_times))
    _append_raw(raw_rows, "Full online pipeline", full_times)

    if not full_times:
        raise RuntimeError(
            "No benchmark frames remained after warm-up; use a longer video, "
            "a lower warm-up count, or a higher max-frame limit"
        )

    _write_dataclass_csv(
        output_directory / "component_benchmark_summary.csv",
        list(summaries),
    )
    _write_dataclass_csv(
        output_directory / "component_benchmark_raw.csv",
        list(raw_rows),
    )
    return summaries
