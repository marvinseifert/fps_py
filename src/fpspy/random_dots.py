"""Randomly placed dots of different sizes, flashed one at a time.

The dot counterpart of `random_bars`: each presentation is a single filled
circle with a random centre and diameter, drawn dark on a light background and
followed by a blank interval. A dot has no orientation, so a diameter is all
that varies besides position.

Like the bars, dot centres are sampled anywhere in the field, so a dot near an
edge is clipped by it, and only two frames are stored per presentation (the
dot, then the blank), each held for its own duration.

`shape_frames` and `append_shape_table` live in `random_bars` - the frame
assembly and the parameter table are the same for both stimuli, and only the
drawing and the sampling differ.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from fpspy.random_bars import append_shape_table, shape_frames  # noqa: F401

# Names of the per-dot arrays produced by sample_dots().
DOT_FIELDS = ("center_x", "center_y", "diameter")


def draw_dot(frame, center_x, center_y, diameter, value) -> bool:
    """Draw one filled circle into a 2D frame, in place.

    Parameters
    ----------
    frame : np.ndarray
        2D (height, width) array, modified in place.
    center_x, center_y : float
        Centre of the dot in pixels, with (0, 0) the top-left corner of the
        frame. May lie outside the frame, in which case the dot is clipped or
        not drawn at all.
    diameter : float
        Diameter of the dot in pixels.
    value : int
        Value written into every covered pixel.

    Returns
    -------
    bool
        Whether any pixel was written, i.e. whether the dot is at least partly
        inside the frame.
    """
    height, width = frame.shape
    radius = diameter / 2.0

    # Only the dot's bounding box is tested, not the whole field, so that
    # generating thousands of small dots on a 1280x800 canvas stays quick.
    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(width, int(math.ceil(center_x + radius)) + 1)
    y0 = max(0, int(math.floor(center_y - radius)))
    y1 = min(height, int(math.ceil(center_y + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return False

    # +0.5 tests the centre of each pixel, so a dot covers about pi*r^2 pixels
    # wherever it lies, rather than depending on how it falls on the grid.
    dx = (np.arange(x0, x1, dtype=np.float64) + 0.5 - center_x)[np.newaxis, :]
    dy = (np.arange(y0, y1, dtype=np.float64) + 0.5 - center_y)[:, np.newaxis]
    mask = (dx * dx + dy * dy) <= radius * radius
    if not mask.any():
        return False
    # Basic slicing gives a view, so the masked assignment lands in `frame`.
    frame[y0:y1, x0:x1][mask] = value
    return True


def sample_dots(
    rng,
    n_dots: int,
    width: int,
    height: int,
    diameter_min: float,
    diameter_max: float,
) -> Dict[str, np.ndarray]:
    """Draw the parameters of `n_dots` random dots.

    Centres are uniform over the whole (width, height) field - not restricted
    to positions where the dot fits - so dots near a border are clipped by it.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness. Pass a seeded generator for a reproducible set.

    Returns
    -------
    dict[str, np.ndarray]
        One (n_dots,) float array per name in `DOT_FIELDS`.
    """
    if n_dots < 1:
        raise ValueError(f"Need at least one dot. {n_dots=}")
    if diameter_min > diameter_max:
        raise ValueError(f"diameter_min > diameter_max. {diameter_min=}, {diameter_max=}")
    if diameter_min <= 0:
        raise ValueError(f"Dot diameter must be positive. {diameter_min=}")

    return {
        "center_x": rng.uniform(0.0, float(width), n_dots),
        "center_y": rng.uniform(0.0, float(height), n_dots),
        "diameter": rng.uniform(float(diameter_min), float(diameter_max), n_dots),
    }


def dot_frames(
    dots: Dict[str, np.ndarray],
    width: int,
    height: int,
    background: int = 255,
    dot_value: int = 0,
    with_blank: bool = True,
) -> np.ndarray:
    """Render sampled dots into a (F, H, W, 1) uint8 frames array."""

    def draw(frame, i):
        draw_dot(
            frame,
            dots["center_x"][i],
            dots["center_y"][i],
            dots["diameter"][i],
            dot_value,
        )

    return shape_frames(
        len(dots["center_x"]),
        draw,
        height=height,
        width=width,
        background=background,
        with_blank=with_blank,
    )
