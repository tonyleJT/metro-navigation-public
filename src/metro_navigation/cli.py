"""Command-line interface for running and benchmarking the prototype."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from metro_navigation.config import Settings


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", help="Inference device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--od-weights", type=Path, help="YOLO metro-object weights")
    parser.add_argument("--segmenter-weights", type=Path, help="SegFormer custom weights")
    parser.add_argument("--sign-weights", type=Path, help="YOLO sign-detector weights")


def _settings_from_arguments(args: argparse.Namespace) -> Settings:
    settings = Settings()
    model_paths = settings.models.with_overrides(
        object_detector=args.od_weights,
        segmenter=args.segmenter_weights,
        sign_detector=args.sign_weights,
    )
    settings = replace(settings, models=model_paths)
    if args.device:
        settings = replace(settings, device=args.device)
    if getattr(args, "station_name", None):
        settings = replace(
            settings,
            fusion=replace(settings.fusion, station_name=args.station_name),
        )
    return settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metro-nav",
        description="Phase-aware computer-vision metro navigation research prototype.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the online/video inference pipeline")
    run_parser.add_argument("--source", required=True, help="Video path or camera index such as 0")
    run_parser.add_argument("--display", action="store_true", help="Show an OpenCV window")
    run_parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech")
    run_parser.add_argument("--output-video", type=Path, help="Optional annotated MP4 output")
    run_parser.add_argument(
        "--runtime-log",
        type=Path,
        default=Path("outputs/runtime_log.csv"),
    )
    run_parser.add_argument(
        "--sample-every",
        type=_non_negative_integer,
        default=0,
        metavar="N",
        help="Save one annotated image every N frames; 0 disables samples",
    )
    run_parser.add_argument(
        "--sample-directory",
        type=Path,
        default=Path("outputs/samples"),
    )
    run_parser.add_argument(
        "--od-every",
        type=_positive_integer,
        default=1,
        metavar="N",
    )
    run_parser.add_argument(
        "--segment-every",
        type=_positive_integer,
        default=1,
        metavar="N",
    )
    run_parser.add_argument(
        "--sign-every",
        type=_positive_integer,
        default=8,
        metavar="N",
    )
    run_parser.add_argument(
        "--station-name",
        help="Optional route-specific station announcement, e.g. 'Tan Cang station'",
    )
    _add_common_model_arguments(run_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark components and the full pipeline on a video",
    )
    benchmark_parser.add_argument("--source", required=True, help="Benchmark video path")
    benchmark_parser.add_argument(
        "--warmup",
        type=_non_negative_integer,
        default=30,
    )
    benchmark_parser.add_argument(
        "--max-frames",
        type=_non_negative_integer,
        default=0,
        help="Maximum frames per pass; 0 means all frames",
    )
    benchmark_parser.add_argument(
        "--sign-every",
        type=_positive_integer,
        default=8,
        metavar="N",
    )
    benchmark_parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/benchmark"),
    )
    _add_common_model_arguments(benchmark_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = _settings_from_arguments(args)

    if args.command == "run":
        from metro_navigation.pipeline import run_video_pipeline

        settings = replace(
            settings,
            runtime=replace(
                settings.runtime,
                object_detection_interval=args.od_every,
                segmentation_interval=args.segment_every,
                sign_detection_interval=args.sign_every,
            ),
        )
        summary = run_video_pipeline(
            source=args.source,
            settings=settings,
            display=args.display,
            output_video=args.output_video,
            sample_interval=args.sample_every,
            sample_directory=args.sample_directory,
            runtime_log=args.runtime_log,
            speech_enabled=not args.no_speech,
        )
        print("\nRuntime summary")
        print(f"  Frames:              {summary['frames']}")
        print(f"  Total time:          {summary['total_seconds']:.3f} s")
        print(f"  Average latency:     {summary['average_ms_per_frame']:.2f} ms/frame")
        print(f"  Average throughput:  {summary['average_fps']:.2f} FPS")
        print(f"  Runtime log:         {summary['runtime_log']}")
        if summary["output_video"]:
            print(f"  Output video:        {summary['output_video']}")
        return 0

    from metro_navigation.benchmark import run_benchmark

    summaries = run_benchmark(
        source=args.source,
        settings=settings,
        warmup_frames=args.warmup,
        max_frames=args.max_frames,
        sign_interval=args.sign_every,
        output_directory=args.output_directory,
    )
    print("\nComponent benchmark")
    print(f"{'Component':34} {'Input':12} {'Latency (ms)':>14} {'FPS':>10}")
    print("-" * 74)
    for row in summaries:
        print(
            f"{row.component:34} {row.input_size:12} "
            f"{row.mean_latency_ms:14.2f} {row.fps_equivalent:10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
