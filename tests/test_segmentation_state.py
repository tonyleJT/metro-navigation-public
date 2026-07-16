from metro_navigation.config import SegmentationSettings
from metro_navigation.domain import SegmentationMode
from metro_navigation.models.segmenter import SegFormerGuidance


def bare_segmenter(settings: SegmentationSettings) -> SegFormerGuidance:
    segmenter = SegFormerGuidance.__new__(SegFormerGuidance)
    segmenter._settings = settings
    segmenter._mode = SegmentationMode.SEARCHING
    segmenter._on_counter = 0
    segmenter._off_counter = 0
    segmenter._last_lookahead_x = None
    return segmenter


def test_mode_switch_requires_confirmed_blindway_contact() -> None:
    segmenter = bare_segmenter(SegmentationSettings(mode_on_frames=2, mode_off_frames=3))

    segmenter._update_mode(True)
    assert segmenter.mode is SegmentationMode.SEARCHING
    segmenter._update_mode(True)
    assert segmenter.mode is SegmentationMode.FOLLOWING


def test_mode_switch_back_requires_confirmed_contact_loss() -> None:
    segmenter = bare_segmenter(SegmentationSettings(mode_on_frames=1, mode_off_frames=2))
    segmenter._update_mode(True)
    assert segmenter.mode is SegmentationMode.FOLLOWING

    segmenter._update_mode(False)
    assert segmenter.mode is SegmentationMode.FOLLOWING
    segmenter._update_mode(False)
    assert segmenter.mode is SegmentationMode.SEARCHING
