from metro_navigation.utils.clock import (
    anchor_line_geometry,
    angle_degrees_to_clock,
    points_to_clock,
)


def test_cardinal_clock_directions() -> None:
    assert angle_degrees_to_clock(0) == "12 o'clock"
    assert angle_degrees_to_clock(90) == "3 o'clock"
    assert angle_degrees_to_clock(180) == "6 o'clock"
    assert angle_degrees_to_clock(-90) == "9 o'clock"


def test_points_to_clock_uses_up_as_twelve() -> None:
    root = (100, 100)
    assert points_to_clock(root, (100, 50))[0] == "12 o'clock"
    assert points_to_clock(root, (150, 100))[0] == "3 o'clock"
    assert points_to_clock(root, (50, 100))[0] == "9 o'clock"


def test_coincident_points_have_no_direction() -> None:
    assert points_to_clock((10, 10), (10, 10)) == (None, 0.0)


def test_anchor_line_stays_inside_frame() -> None:
    start, end, root, thickness = anchor_line_geometry(640, 480)
    assert 0 <= start[0] <= end[0] < 640
    assert start[1] == end[1] == root[1]
    assert root[0] == 320
    assert thickness > 0
