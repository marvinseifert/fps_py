"""Randomly placed, sized and oriented bars, flashed one at a time.

Each presentation is a single bar — a filled rectangle with a random centre,
length, thickness and orientation — drawn dark on a light background, followed
by a blank (background-only) interval before the next bar.

Only two distinct images exist per presentation (the bar, then the blank), so
the stimulus is built as two stored frames per bar with per-frame durations,
rather than one stored frame per displayed frame. A 0.5 s bar therefore costs
one 1280x800 frame, not 30 of them; the presenter holds each stored frame for
its own duration (see `StimArray`'s `frame_durations`).

A bar's centre is sampled anywhere in the field, including near the edges, so
bars can be clipped by the frame border and the visible bar area varies
between presentations.

The array convention is the one the rest of the pipeline uses: (0, 0) is the
top-left corner, and frames are (F, H, W, C).
"""

from __future__ import annotations

import math
from typing import Dict

import h5py
import numpy as np

# Names of the per-bar arrays produced by sample_bars().
BAR_FIELDS = ("center_x", "center_y", "length", "thickness", "angle_deg")


def draw_bar(frame, center_x, center_y, length, thickness, angle_deg, value) -> bool:
    """Draw one filled, rotated rectangle into a 2D frame, in place.

    Parameters
    ----------
    frame : np.ndarray
        2D (height, width) array, modified in place.
    center_x, center_y : float
        Centre of the bar in pixels, with (0, 0) the top-left corner of the
        frame. May lie outside the frame, in which case the bar is clipped or
        not drawn at all.
    length : float
        Extent of the bar along its long axis, in pixels.
    thickness : float
        Extent of the bar perpendicular to its long axis, in pixels.
    angle_deg : float
        Orientation of the long axis. 0 is horizontal. Because y grows
        downwards (top-left origin), increasing angles turn the bar clockwise
        as it appears on the display. A bar is symmetric, so only the angle
        modulo 180 matters.
    value : int
        Value written into every covered pixel.

    Returns
    -------
    bool
        Whether any pixel was written, i.e. whether the bar is at least partly
        inside the frame.
    """
    height, width = frame.shape
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_length, half_thickness = length / 2.0, thickness / 2.0

    # Axis-aligned bounding box of the rotated rectangle. Evaluating the mask
    # only here, rather than over the whole 1280x800 field once per bar, is
    # what keeps generating a few thousand bars quick.
    extent_x = half_length * abs(cos_t) + half_thickness * abs(sin_t)
    extent_y = half_length * abs(sin_t) + half_thickness * abs(cos_t)
    x0 = max(0, int(math.floor(center_x - extent_x)))
    x1 = min(width, int(math.ceil(center_x + extent_x)) + 1)
    y0 = max(0, int(math.floor(center_y - extent_y)))
    y1 = min(height, int(math.ceil(center_y + extent_y)) + 1)
    if x0 >= x1 or y0 >= y1:
        return False

    # +0.5 tests the centre of each pixel, so a bar of thickness t covers
    # about t pixels wherever it lies, rather than depending on how its edges
    # happen to fall relative to the integer grid.
    dx = (np.arange(x0, x1, dtype=np.float64) + 0.5 - center_x)[np.newaxis, :]
    dy = (np.arange(y0, y1, dtype=np.float64) + 0.5 - center_y)[:, np.newaxis]
    along = dx * cos_t + dy * sin_t
    across = -dx * sin_t + dy * cos_t
    mask = (np.abs(along) <= half_length) & (np.abs(across) <= half_thickness)
    if not mask.any():
        return False
    # Basic slicing gives a view, so the masked assignment lands in `frame`.
    frame[y0:y1, x0:x1][mask] = value
    return True


def sample_bars(
    rng,
    n_bars: int,
    width: int,
    height: int,
    length_min: float,
    length_max: float,
    thickness_min: float,
    thickness_max: float,
    n_orientations: int,
) -> Dict[str, np.ndarray]:
    """Draw the parameters of `n_bars` random bars.

    Centres are uniform over the whole (width, height) field — not restricted
    to positions where the bar fits — so bars near a border are clipped by it.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness. Pass a seeded generator for a reproducible set.
    n_orientations : int
        0 samples the orientation uniformly from [0, 180). Any positive value
        instead picks uniformly among that many equally spaced orientations,
        starting at 0 (e.g. 4 gives 0, 45, 90 and 135 degrees).

    Returns
    -------
    dict[str, np.ndarray]
        One (n_bars,) float array per name in `BAR_FIELDS`.
    """
    if n_bars < 1:
        raise ValueError(f"Need at least one bar. {n_bars=}")
    if length_min > length_max:
        raise ValueError(f"length_min > length_max. {length_min=}, {length_max=}")
    if thickness_min > thickness_max:
        raise ValueError(
            f"thickness_min > thickness_max. {thickness_min=}, {thickness_max=}"
        )
    if length_min <= 0 or thickness_min <= 0:
        raise ValueError(
            f"Bar dimensions must be positive. {length_min=}, {thickness_min=}"
        )

    if n_orientations > 0:
        step = 180.0 / n_orientations
        angle_deg = rng.integers(0, n_orientations, n_bars) * step
    else:
        angle_deg = rng.uniform(0.0, 180.0, n_bars)

    return {
        "center_x": rng.uniform(0.0, float(width), n_bars),
        "center_y": rng.uniform(0.0, float(height), n_bars),
        "length": rng.uniform(float(length_min), float(length_max), n_bars),
        "thickness": rng.uniform(float(thickness_min), float(thickness_max), n_bars),
        "angle_deg": np.asarray(angle_deg, dtype=np.float64),
    }


def shape_frames(
    n_shapes: int,
    draw,
    height: int,
    width: int,
    background: int = 255,
    with_blank: bool = True,
) -> np.ndarray:
    """Assemble a (F, H, W, 1) uint8 frames array, one shape per frame.

    Shared by every "one random shape at a time" stimulus: the only thing that
    differs between them is what gets drawn.

    Parameters
    ----------
    draw : callable
        ``draw(frame, i)`` draws shape `i` into a 2D (height, width) view of
        the frame it is shown in, in place.
    with_blank : bool
        Follow each shape with a blank background frame.

    The channel dimension is 1: `StimArray` broadcasts it to however many
    channels the display has.
    """
    per_shape = 2 if with_blank else 1
    frames = np.full(
        (n_shapes * per_shape, height, width, 1), background, dtype=np.uint8
    )
    for i in range(n_shapes):
        draw(frames[i * per_shape, :, :, 0], i)
    return frames


def bar_frames(
    bars: Dict[str, np.ndarray],
    width: int,
    height: int,
    background: int = 255,
    bar_value: int = 0,
    with_blank: bool = True,
) -> np.ndarray:
    """Render sampled bars into a (F, H, W, 1) uint8 frames array."""

    def draw(frame, i):
        draw_bar(
            frame,
            bars["center_x"][i],
            bars["center_y"][i],
            bars["length"][i],
            bars["thickness"][i],
            bars["angle_deg"][i],
            bar_value,
        )

    return shape_frames(
        len(bars["center_x"]),
        draw,
        height=height,
        width=width,
        background=background,
        with_blank=with_blank,
    )


def append_shape_table(path, shapes: Dict[str, np.ndarray], per_shape: int, group: str):
    """Append the per-shape parameters to an already-written stimulus file.

    Without this, the only record of what was shown when is the seed. The
    table goes into its own group; the v1 reader looks up its own keys by
    name, so the extra group is ignored on load and playback.

    `frame_index` is the index into the stimulus' frames array at which each
    shape is shown, which is also the index of its trigger.
    """
    n_shapes = len(next(iter(shapes.values())))
    with h5py.File(path, "a") as f:
        h5_group = f.create_group(group)
        h5_group.create_dataset(
            "frame_index", data=np.arange(n_shapes) * per_shape, dtype="uint64"
        )
        for name, values in shapes.items():
            h5_group.create_dataset(name, data=np.asarray(values), dtype="float64")
