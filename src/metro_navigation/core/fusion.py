"""Phase-aware fusion of object, sign, and segmentation outputs."""

from __future__ import annotations

import time
from collections.abc import Callable

from metro_navigation.config import FusionSettings
from metro_navigation.domain import (
    GuidanceDecision,
    NavigationPhase,
    ObjectDetection,
    ObjectDetectionResult,
    Point,
    SegmentationMode,
    SegmentationResult,
    SignDetectionResult,
    SignPair,
    SpeakerProtocol,
)
from metro_navigation.utils.clock import points_to_clock


class NavigationFusion:
    """Convert perception results into one stable guidance decision per frame.

    The state machine keeps route phases monotonic, recovers to a later phase
    from strong scene evidence, and emits at most one speech message per frame.
    """

    def __init__(
        self,
        speaker: SpeakerProtocol,
        settings: FusionSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._speaker = speaker
        self._settings = settings
        self._clock = clock

        self.phase = NavigationPhase.ENTRY
        self.segmentation_enabled = True
        self._escalator_was_visible = False

        self._last_ticket_booth_time = 0.0
        self._saw_ticket_booth = False
        self._last_destination_repeat = 0.0

        self._last_guidance_time = float("-inf")
        self._last_guidance_text = ""
        self._ui_text = ""
        self._spoken_events: set[str] = set()
        self._speech_emitted_this_frame = False

        self._stage_two_turn_left_done = False
        self._stage_two_straight_only = False
        self._stage_two_gate_seen_time = 0.0
        self._stage_two_gate_direction_known = False

        self._candidate_phase = self.phase
        self._candidate_phase_count = 0

    @property
    def ui_text(self) -> str:
        return self._ui_text

    def _find_pair(
        self,
        sign_result: SignDetectionResult,
        pair_type: str,
    ) -> SignPair | None:
        return next(
            (
                pair
                for platform in sign_result.platforms
                for pair in platform.pairs
                if pair.pair_type == pair_type
            ),
            None,
        )

    @staticmethod
    def _best_detection(
        result: ObjectDetectionResult,
        class_name: str,
    ) -> ObjectDetection | None:
        return max(
            (detection for detection in result.detections if detection.class_name == class_name),
            key=lambda detection: detection.confidence,
            default=None,
        )

    def _speak(
        self,
        text: str,
        *,
        now: float,
        force: bool = False,
        event_key: str | None = None,
    ) -> bool:
        """Emit at most one accepted speech item per update call."""

        if not text or self._speech_emitted_this_frame:
            return False
        if event_key is not None and event_key in self._spoken_events:
            return False
        if not self._speaker.say(text, force=force):
            return False

        self._speech_emitted_this_frame = True
        self._ui_text = text
        self._last_guidance_time = now
        self._last_guidance_text = text
        if event_key is not None:
            self._spoken_events.add(event_key)
        return True

    def _guidance_is_due(self, text: str, now: float) -> bool:
        interval = (
            self._settings.repeated_guidance_interval_seconds
            if text == self._last_guidance_text
            else self._settings.changed_guidance_cooldown_seconds
        )
        return now - self._last_guidance_time >= interval

    def _speak_guidance(
        self,
        text: str,
        *,
        now: float,
        force_to_speaker: bool = False,
    ) -> bool:
        if not self._guidance_is_due(text, now):
            return False
        return self._speak(text, now=now, force=force_to_speaker)

    def _apply_direction_filter(
        self,
        root_point: Point,
        target_point: Point,
    ) -> tuple[bool, str | None, float]:
        clock_direction, angle_degrees = points_to_clock(root_point, target_point)
        if clock_direction is None:
            return False, None, angle_degrees

        if (
            self.phase is NavigationPhase.AFTER_FIRST_ESCALATOR
            and self._stage_two_straight_only
            and abs(angle_degrees) > self._settings.stage_two_straight_angle_degrees
        ):
            return False, clock_direction, angle_degrees

        if (
            self.phase
            in {
                NavigationPhase.AFTER_TICKET_GATE,
                NavigationPhase.FINAL_PLATFORM,
            }
            and angle_degrees > 0.0
        ):
            return False, clock_direction, angle_degrees

        return True, clock_direction, angle_degrees

    def _infer_scene_phase(
        self,
        object_result: ObjectDetectionResult,
        sign_result: SignDetectionResult,
    ) -> NavigationPhase:
        stair = self._best_detection(object_result, "stair node")
        ticket_booth = self._best_detection(object_result, "ticket booth")
        no_entry_gate = self._find_pair(sign_result, "no-entry_gate")
        ben_thanh_suoi_tien = self._find_pair(
            sign_result,
            "ben-thanh_suoi-tien",
        )
        has_platform_sign = sign_result.has_platform_sign
        has_ben_thanh_in_platform = any(
            component.class_name == "ben-thanh-station"
            for platform in sign_result.platforms
            for component in platform.components
        )

        if has_platform_sign and has_ben_thanh_in_platform:
            return NavigationPhase.FINAL_PLATFORM
        if ben_thanh_suoi_tien is not None:
            return NavigationPhase.AFTER_TICKET_GATE
        if (
            no_entry_gate is not None
            or sign_result.has_ticket_sign
            or sign_result.has_gate
            or ticket_booth is not None
        ):
            return NavigationPhase.AFTER_FIRST_ESCALATOR
        if stair is not None:
            return NavigationPhase.ENTRY
        return self.phase

    def _update_scene_phase(self, inferred_phase: NavigationPhase) -> None:
        """Confirm forward phase changes and reject route regressions."""

        if inferred_phase <= self.phase:
            self._candidate_phase = self.phase
            self._candidate_phase_count = 0
            return

        if inferred_phase == self._candidate_phase:
            self._candidate_phase_count += 1
        else:
            self._candidate_phase = inferred_phase
            self._candidate_phase_count = 1

        if self._candidate_phase_count >= self._settings.phase_confirmation_frames:
            self.phase = inferred_phase
            self._candidate_phase_count = 0

    @staticmethod
    def _segmentation_candidate(result: SegmentationResult) -> Point | None:
        return (
            result.lookahead_target
            if result.mode is SegmentationMode.FOLLOWING
            else result.safe_target
        )

    def _decision_for_target(
        self,
        segmentation_result: SegmentationResult,
        target_point: Point | None,
        *,
        now: float,
        verb: str,
        apply_filter: bool = True,
    ) -> GuidanceDecision:
        if target_point is None:
            return GuidanceDecision(ui_text=self._ui_text)

        if apply_filter:
            allowed, clock_direction, angle_degrees = self._apply_direction_filter(
                segmentation_result.root_point,
                target_point,
            )
        else:
            clock_direction, angle_degrees = points_to_clock(
                segmentation_result.root_point,
                target_point,
            )
            allowed = clock_direction is not None

        if not allowed or clock_direction is None:
            return GuidanceDecision(ui_text=self._ui_text)

        self._speak_guidance(
            f"{verb} {clock_direction}",
            now=now,
        )
        return GuidanceDecision(
            target_point=target_point,
            ui_text=self._ui_text,
            clock_direction=clock_direction,
            angle_degrees=angle_degrees,
        )

    def update(
        self,
        object_result: ObjectDetectionResult,
        segmentation_result: SegmentationResult,
        sign_result: SignDetectionResult,
    ) -> GuidanceDecision:
        """Update state from one frame of perception results."""

        now = self._clock()
        self._speech_emitted_this_frame = False
        self._update_scene_phase(self._infer_scene_phase(object_result, sign_result))

        ticket_booth = self._best_detection(object_result, "ticket booth")
        if ticket_booth is not None:
            self._last_ticket_booth_time = now
            self._saw_ticket_booth = True

        if object_result.escalator_detections:
            escalator = object_result.escalator_detections[0]
            self.segmentation_enabled = False
            self._escalator_was_visible = True
            if escalator.clock_direction:
                self._speak_guidance(
                    f"Escalator at {escalator.clock_direction}",
                    now=now,
                    force_to_speaker=True,
                )
            return GuidanceDecision(
                target_point=escalator.center,
                ui_text=self._ui_text,
                clock_direction=escalator.clock_direction,
                angle_degrees=escalator.angle_degrees,
            )

        if self._escalator_was_visible:
            self._escalator_was_visible = False
            self.segmentation_enabled = True

        if self.phase is NavigationPhase.ENTRY:
            return self._update_entry(
                object_result,
                segmentation_result,
                now=now,
            )
        if self.phase is NavigationPhase.AFTER_FIRST_ESCALATOR:
            return self._update_after_first_escalator(
                object_result,
                segmentation_result,
                sign_result,
                now=now,
            )
        if self.phase is NavigationPhase.AFTER_TICKET_GATE:
            return self._update_after_ticket_gate(
                object_result,
                segmentation_result,
                sign_result,
                now=now,
            )
        return self._update_final_platform(
            object_result,
            segmentation_result,
            sign_result,
            now=now,
        )

    def _update_entry(
        self,
        object_result: ObjectDetectionResult,
        segmentation_result: SegmentationResult,
        *,
        now: float,
    ) -> GuidanceDecision:
        stair = self._best_detection(object_result, "stair node")
        if (
            stair is not None
            and self._settings.station_name
            and "entry_station_name" not in self._spoken_events
        ):
            self._speak(
                self._settings.station_name,
                now=now,
                force=True,
                event_key="entry_station_name",
            )

        if stair is not None and segmentation_result.standalone_curb_target is not None:
            return self._decision_for_target(
                segmentation_result,
                segmentation_result.standalone_curb_target,
                now=now,
                verb="Follow",
            )
        if stair is not None:
            if stair.clock_direction:
                self._speak_guidance(
                    f"Move {stair.clock_direction}",
                    now=now,
                )
            return GuidanceDecision(
                target_point=stair.center,
                ui_text=self._ui_text,
                clock_direction=stair.clock_direction,
                angle_degrees=stair.angle_degrees,
            )

        candidate = self._segmentation_candidate(segmentation_result)
        verb = "Follow" if segmentation_result.mode is SegmentationMode.FOLLOWING else "Move"
        return self._decision_for_target(
            segmentation_result,
            candidate,
            now=now,
            verb=verb,
        )

    def _update_after_first_escalator(
        self,
        object_result: ObjectDetectionResult,
        segmentation_result: SegmentationResult,
        sign_result: SignDetectionResult,
        *,
        now: float,
    ) -> GuidanceDecision:
        self.segmentation_enabled = True
        gate_pair = self._find_pair(sign_result, "no-entry_gate")
        if gate_pair is not None:
            self._stage_two_gate_direction_known = True
            if "stage_two_gate_direction" not in self._spoken_events:
                self._stage_two_gate_seen_time = now
                if gate_pair.left_label == "no-entry" and gate_pair.right_label == "gate":
                    message = "Ticket gate is on the right"
                elif gate_pair.left_label == "gate" and gate_pair.right_label == "no-entry":
                    message = "Ticket gate is on the left"
                else:
                    message = "Ticket gate ahead"
                self._speak(
                    message,
                    now=now,
                    force=True,
                    event_key="stage_two_gate_direction",
                )

        target = None
        if (
            not self._stage_two_turn_left_done
            and segmentation_result.right_edge_curb_target is not None
        ):
            target = segmentation_result.right_edge_curb_target
            turn_is_armed = (
                self._stage_two_gate_direction_known
                and self._stage_two_gate_seen_time > 0.0
                and now - self._stage_two_gate_seen_time
                >= self._settings.stage_two_turn_delay_seconds
            )
            if (
                turn_is_armed
                and segmentation_result.right_edge_curb_close
                and "stage_two_turn_left" not in self._spoken_events
                and self._speak(
                    "Turn left",
                    now=now,
                    force=True,
                    event_key="stage_two_turn_left",
                )
            ):
                self._stage_two_turn_left_done = True
                self._stage_two_straight_only = True

        if (
            sign_result.has_ticket_sign
            and sign_result.has_gate
            and "stage_two_gate_ahead" not in self._spoken_events
            and self._speak(
                "Gate ahead",
                now=now,
                force=True,
                event_key="stage_two_gate_ahead",
            )
        ):
            self._stage_two_straight_only = True
        elif sign_result.has_ticket_sign and "stage_two_ticket_counter" not in self._spoken_events:
            self._speak(
                "Ticket counter on the right",
                now=now,
                force=True,
                event_key="stage_two_ticket_counter",
            )

        if self._stage_two_straight_only or target is None:
            target = self._segmentation_candidate(segmentation_result)

        if self._saw_ticket_booth and (
            now - self._last_ticket_booth_time > self._settings.ticket_booth_pass_seconds
        ):
            self.phase = NavigationPhase.AFTER_TICKET_GATE
            self._stage_two_straight_only = False

        verb = "Follow" if segmentation_result.mode is SegmentationMode.FOLLOWING else "Move"
        return self._decision_for_target(
            segmentation_result,
            target,
            now=now,
            verb=verb,
        )

    def _update_after_ticket_gate(
        self,
        object_result: ObjectDetectionResult,
        segmentation_result: SegmentationResult,
        sign_result: SignDetectionResult,
        *,
        now: float,
    ) -> GuidanceDecision:
        self.segmentation_enabled = True
        destination_pair = self._find_pair(
            sign_result,
            "ben-thanh_suoi-tien",
        )
        if destination_pair is not None and (
            "destination_direction" not in self._spoken_events
            or now - self._last_destination_repeat >= self._settings.destination_repeat_seconds
        ):
            side = "left" if destination_pair.left_label == "ben-thanh-station" else "right"
            if self._speak(
                f"Ben Thanh station is on the {side} turn",
                now=now,
                force=True,
            ):
                self._spoken_events.add("destination_direction")
                self._last_destination_repeat = now

        stair = self._best_detection(object_result, "stair node")
        if (
            stair is not None
            and stair.clock_direction
            and "stage_three_stairs" not in self._spoken_events
        ):
            self._speak(
                f"Stairs at {stair.clock_direction}",
                now=now,
                force=True,
                event_key="stage_three_stairs",
            )

        target: tuple[int, int] | None
        if stair is not None:
            target = segmentation_result.standalone_curb_target or stair.center
        else:
            target = self._segmentation_candidate(segmentation_result)
        verb = "Follow" if segmentation_result.mode is SegmentationMode.FOLLOWING else "Move"
        return self._decision_for_target(
            segmentation_result,
            target,
            now=now,
            verb=verb,
        )

    def _update_final_platform(
        self,
        object_result: ObjectDetectionResult,
        segmentation_result: SegmentationResult,
        sign_result: SignDetectionResult,
        *,
        now: float,
    ) -> GuidanceDecision:
        self.segmentation_enabled = True

        closed_gate = next(
            (
                detection
                for detection in object_result.detections
                if detection.class_name == "closed gate"
                and detection.clock_direction == "12 o'clock"
            ),
            None,
        )
        open_gate = next(
            (
                detection
                for detection in object_result.detections
                if detection.class_name == "open gate" and detection.clock_direction == "12 o'clock"
            ),
            None,
        )
        if closed_gate is not None:
            self._speak(
                "Door is closing. Wait for the train.",
                now=now,
                force=True,
                event_key="final_closed_gate",
            )
        elif open_gate is not None:
            self._speak(
                "The door is open. Please get on the train.",
                now=now,
                force=True,
                event_key="final_open_gate",
            )

        has_platform_sign = sign_result.has_platform_sign
        has_ben_thanh_in_platform = any(
            component.class_name == "ben-thanh-station"
            for platform in sign_result.platforms
            for component in platform.components
        )
        if has_platform_sign and has_ben_thanh_in_platform:
            self._speak(
                "Turn left to get on the train",
                now=now,
                force=True,
                event_key="final_turn_left_train",
            )

        target = segmentation_result.left_turn_curb_target or self._segmentation_candidate(
            segmentation_result
        )
        verb = "Follow" if segmentation_result.mode is SegmentationMode.FOLLOWING else "Move"
        return self._decision_for_target(
            segmentation_result,
            target,
            now=now,
            verb=verb,
        )
