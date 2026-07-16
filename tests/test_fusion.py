from __future__ import annotations

from dataclasses import replace

from metro_navigation.config import FusionSettings
from metro_navigation.core.fusion import NavigationFusion
from metro_navigation.domain import (
    NavigationPhase,
    ObjectDetection,
    ObjectDetectionResult,
    SegmentationMode,
    SegmentationResult,
    SignComponent,
    SignDetectionResult,
    SignPair,
    SignPlatform,
)


class RecordingSpeaker:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def say(self, text: str, *, force: bool = False) -> bool:
        self.messages.append((text, force))
        return True

    def stop(self) -> None:
        return None


class FakeClock:
    def __init__(self, current: float = 0.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current


def segmentation_result(
    *,
    mode: SegmentationMode = SegmentationMode.SEARCHING,
    safe_target: tuple[int, int] | None = (320, 300),
    lookahead_target: tuple[int, int] | None = None,
) -> SegmentationResult:
    return SegmentationResult(
        mask_bgr=None,
        line_start=(250, 474),
        line_end=(390, 474),
        root_point=(320, 474),
        line_thickness=10,
        mode=mode,
        safe_target=safe_target,
        lookahead_target=lookahead_target,
    )


def detection(
    class_name: str,
    *,
    center: tuple[int, int] = (320, 200),
    confidence: float = 0.9,
    clock_direction: str = "12 o'clock",
) -> ObjectDetection:
    return ObjectDetection(
        class_name=class_name,
        confidence=confidence,
        bbox=(300, 180, 340, 220),
        center=center,
        clock_direction=clock_direction,
        angle_degrees=0.0,
    )


def test_entry_falls_back_to_segmentation_when_no_landmark_is_visible() -> None:
    speaker = RecordingSpeaker()
    fusion = NavigationFusion(speaker, FusionSettings(), clock=FakeClock())

    decision = fusion.update(
        ObjectDetectionResult(),
        segmentation_result(),
        SignDetectionResult(),
    )

    assert decision.target_point == (320, 300)
    assert decision.clock_direction == "12 o'clock"
    assert speaker.messages == [("Move 12 o'clock", False)]


def test_same_guidance_repeats_only_after_repeat_interval() -> None:
    speaker = RecordingSpeaker()
    clock = FakeClock()
    settings = replace(
        FusionSettings(),
        changed_guidance_cooldown_seconds=4.0,
        repeated_guidance_interval_seconds=8.0,
    )
    fusion = NavigationFusion(speaker, settings, clock=clock)
    inputs = (
        ObjectDetectionResult(),
        segmentation_result(),
        SignDetectionResult(),
    )

    fusion.update(*inputs)
    clock.current = 4.0
    fusion.update(*inputs)
    clock.current = 8.0
    fusion.update(*inputs)

    assert [message for message, _ in speaker.messages] == [
        "Move 12 o'clock",
        "Move 12 o'clock",
    ]


def test_phase_inference_is_forward_only() -> None:
    speaker = RecordingSpeaker()
    fusion = NavigationFusion(
        speaker,
        replace(FusionSettings(), phase_confirmation_frames=2),
        clock=FakeClock(),
    )
    destination_pair = SignPair(
        pair_type="ben-thanh_suoi-tien",
        left_label="ben-thanh-station",
        right_label="suoi-tien-station",
        left_center=(100.0, 100.0),
        right_center=(200.0, 100.0),
    )
    destination_sign = SignDetectionResult(
        platforms=(
            SignPlatform(
                bbox=(0, 0, 300, 200),
                pairs=[destination_pair],
            ),
        )
    )

    fusion.update(ObjectDetectionResult(), segmentation_result(), destination_sign)
    fusion.update(ObjectDetectionResult(), segmentation_result(), destination_sign)
    assert fusion.phase is NavigationPhase.AFTER_TICKET_GATE

    lower_phase_sign = SignDetectionResult(has_ticket_sign=True)
    fusion.update(ObjectDetectionResult(), segmentation_result(), lower_phase_sign)
    fusion.update(ObjectDetectionResult(), segmentation_result(), lower_phase_sign)
    assert fusion.phase is NavigationPhase.AFTER_TICKET_GATE


def test_virtual_platform_does_not_trigger_final_platform_phase() -> None:
    speaker = RecordingSpeaker()
    fusion = NavigationFusion(
        speaker,
        replace(FusionSettings(), phase_confirmation_frames=1),
        clock=FakeClock(),
    )
    virtual_platform = SignPlatform(
        bbox=(0, 0, 640, 480),
        components=[
            SignComponent(
                class_name="ben-thanh-station",
                confidence=0.95,
                bbox=(10, 10, 100, 50),
                center=(55.0, 30.0),
            )
        ],
    )

    fusion.update(
        ObjectDetectionResult(),
        segmentation_result(),
        SignDetectionResult(
            platforms=(virtual_platform,),
            has_platform_sign=False,
            has_ben_thanh=True,
        ),
    )
    assert fusion.phase is NavigationPhase.ENTRY


def test_segmentation_reenabled_after_escalator_disappears() -> None:
    speaker = RecordingSpeaker()
    fusion = NavigationFusion(speaker, FusionSettings(), clock=FakeClock())
    escalator = detection("escalator entry node")

    fusion.update(
        ObjectDetectionResult(
            detections=(escalator,),
            escalator_detections=(escalator,),
        ),
        segmentation_result(),
        SignDetectionResult(),
    )
    assert fusion.segmentation_enabled is False

    fusion.update(
        ObjectDetectionResult(),
        segmentation_result(),
        SignDetectionResult(),
    )
    assert fusion.segmentation_enabled is True


def test_only_one_speech_message_is_emitted_per_frame() -> None:
    speaker = RecordingSpeaker()
    fusion = NavigationFusion(speaker, FusionSettings(), clock=FakeClock())
    fusion.phase = NavigationPhase.AFTER_FIRST_ESCALATOR
    gate_pair = SignPair(
        pair_type="no-entry_gate",
        left_label="no-entry",
        right_label="gate",
        left_center=(100.0, 100.0),
        right_center=(200.0, 100.0),
    )
    sign_result = SignDetectionResult(
        platforms=(SignPlatform(bbox=(0, 0, 300, 200), pairs=[gate_pair]),),
        has_ticket_sign=True,
        has_gate=True,
    )

    fusion.update(
        ObjectDetectionResult(),
        segmentation_result(),
        sign_result,
    )

    assert len(speaker.messages) == 1
    assert speaker.messages[0][0] == "Ticket gate is on the right"
