"""Geometry helpers for clock-face navigation instructions."""

from __future__ import annotations

import math

from metro_navigation.domain import Point

CLOCK_LABELS = {
    0: "12 o'clock",
    1: "1 o'clock",
    2: "2 o'clock",
    3: "3 o'clock",
    4: "4 o'clock",
    5: "5 o'clock",
    6: "6 o'clock",
    7: "7 o'clock",
    8: "8 o'clock",
    9: "9 o'clock",
    10: "10 o'clock",
    11: "11 o'clock",
}


def angle_degrees_to_clock(angle_degrees: float) -> str:
    """Convert a signed image-plane heading to the nearest clock direction.

    Zero degrees points upward. Positive angles rotate clockwise (right), and
    negative angles rotate counter-clockwise (left).
    """

    normalized = angle_degrees % 360.0
    hour = int((normalized + 15.0) // 30.0) % 12
    return CLOCK_LABELS[hour]


def points_to_clock(root_point: Point, target_point: Point) -> tuple[str | None, float]:
    """Return clock direction and signed angle from root to target."""

    root_x, root_y = root_point
    target_x, target_y = target_point
    delta_x = target_x - root_x
    delta_y = root_y - target_y

    if delta_x == 0 and delta_y == 0:
        return None, 0.0

    angle_degrees = math.degrees(math.atan2(delta_x, delta_y))
    return angle_degrees_to_clock(angle_degrees), angle_degrees


def anchor_line_geometry(
    width: int,
    height: int,
    *,
    length_ratio: float = 0.23,
    thickness_pixels: int = 10,
    vertical_offset_pixels: int = 6,
) -> tuple[Point, Point, Point, int]:
    """Return the foot-anchor line and root point for an image frame."""

    root_x = width // 2
    root_y = max(0, height - vertical_offset_pixels)
    half_length = max(1, int(width * length_ratio / 2.0))
    start = (max(0, root_x - half_length), root_y)
    end = (min(width - 1, root_x + half_length), root_y)
    return start, end, (root_x, root_y), max(1, thickness_pixels)
