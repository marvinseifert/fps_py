"""Registry of stimulus generators for the GUI's Generation tab.

Each entry describes a small form (one widget per parameter) and a
``generate()`` function that turns the submitted values into a stimulus file
inside the user's stim folder. Adding a new stimulus template means adding
one ``@register``ed `Generator` here — `gui_qt.py`'s Generation tab iterates
`REGISTRY` and builds the form from it, so it does not need to change to pick
up a new template.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

import fpspy.create_noise
import fpspy.random_bars
import fpspy.random_dots
import fpspy.shuffle_noise
import fpspy.stim


@dataclasses.dataclass(frozen=True)
class Param:
    """One editable field in a generator's form."""

    name: str  # keyword argument name passed to generate()
    label: str  # shown next to the field in the GUI
    kind: str  # "int" | "float" | "bool" | "choice"
    default: Union[int, float, bool]
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    tooltip: str = ""
    # "choice" only: the field's allowed values depend on another field's
    # current value. depends_on names that field; choices_fn(dependency's
    # current value) -> the list of currently-valid values. The GUI
    # re-derives the choice list live as the dependency changes, so an
    # invalid combination can't be entered in the first place.
    depends_on: Optional[str] = None
    choices_fn: Optional[Callable[[object], list]] = None


@dataclasses.dataclass(frozen=True)
class Generator:
    key: str  # stable id; used as the default file name
    name: str  # shown as the form's group box title
    params: tuple  # tuple[Param, ...]
    # generate(stim_dir, file_name, **param_values) -> path of the file written
    generate: Callable[..., Path]


REGISTRY: dict = {}  # dict[str, Generator]


def register(generator: Generator) -> Generator:
    REGISTRY[generator.key] = generator
    return generator


def _h5_name(file_name: str) -> str:
    """Force a .h5 suffix, the way the old GUI's on_generate_noise did."""
    stem = Path(file_name).stem or "stimulus"
    return f"{stem}.h5"


# --- BW checkerboard noise --------------------------------------------------


def _generate_bw_noise(
    stim_dir: Path,
    file_name: str,
    checker_size: int,
    width: int,
    height: int,
    frequency: float,
    duration: float,
    shuffle: bool,
    shuffle_step: Optional[int],
) -> Path:
    path = stim_dir / _h5_name(file_name)
    # Pattern-change rate (frequency) times duration is the number of distinct
    # patterns needed; each is stored once and played at `frequency` fps. See
    # GUI_workplan.md — the old tkinter GUI multiplied by an extra, wrong 60.
    n_frames = max(1, round(duration * frequency))

    if shuffle:
        if shuffle_step is None:
            raise ValueError(
                f"No valid shuffle step for checker_size={checker_size} "
                "(checker must be at least 3px to shuffle at all)."
            )
        # A sub-cell roll (shuffle_pattern) breaks the "one uniform value per
        # checker cell" assumption zoom relies on — after the roll, a cell
        # can straddle two different original values — so this can't be
        # stored zoomed; it needs the full pre-expanded resolution.
        frames = np.stack(
            [
                fpspy.shuffle_noise.shuffle_pattern(
                    fpspy.create_noise.checkerboard(checker_size, width, height),
                    checker_size,
                    shuffle_step,
                )
                for _ in range(n_frames)
            ],
            axis=0,
        )
        zoom = 1
        label = f"BW checkerboard noise, {checker_size}px, shuffled (step={shuffle_step})"
    else:
        # Not shuffled: every checker cell is one uniform value, so one
        # array element per cell plus zoom=checker_size reconstructs the
        # same image at display time, for a much smaller file.
        frames = np.stack(
            [
                fpspy.create_noise.checkerboard_lowres(checker_size, width, height)
                for _ in range(n_frames)
            ],
            axis=0,
        )
        zoom = checker_size
        label = f"BW checkerboard noise, {checker_size}px"

    frames = frames[..., np.newaxis]  # (F, H, W) -> (F, H, W, 1): broadcasts to all channels.

    stim_array = fpspy.stim.StimArray(
        frames,
        frame_durations=1.0 / frequency,
        zoom=zoom,
        triggers=None,
        label=label,
        metadata={
            "checker_size": checker_size,
            "shuffle": shuffle,
            **({"shuffle_step": shuffle_step} if shuffle else {}),
        },
    )
    stim_array.write_hdf5(path)
    return path


BW_NOISE = register(
    Generator(
        key="bw_noise",
        name="Black & white checkerboard noise",
        params=(
            Param(
                "checker_size", "Checker size (px)", "int", 20, minimum=1, maximum=2000
            ),
            Param("width", "Width (px)", "int", 1000, minimum=1, maximum=10000),
            Param("height", "Height (px)", "int", 1000, minimum=1, maximum=10000),
            Param(
                "frequency",
                "Pattern rate (Hz)",
                "float",
                20.0,
                minimum=0.1,
                maximum=1000.0,
            ),
            Param(
                "duration", "Duration (s)", "float", 5.0, minimum=0.1, maximum=36000.0
            ),
            Param(
                "shuffle",
                "Shuffle (jitter each pattern)",
                "bool",
                False,
                tooltip="Roll each pattern by a random sub-cell offset before "
                "storing it, instead of an independent random pattern.",
            ),
            Param(
                "shuffle_step",
                "Shuffle step (px)",
                "choice",
                1,
                tooltip="Spacing between possible shuffle positions. Must "
                "evenly divide (checker size - 2): positions start at 0, "
                "stay evenly spaced, and never let a neighbouring checker "
                "cover half or more of this one.",
                depends_on="checker_size",
                choices_fn=fpspy.shuffle_noise.valid_shuffle_steps,
            ),
        ),
        generate=_generate_bw_noise,
    )
)


# --- Moving bar (shader-based) -----------------------------------------------
# Delegates to the existing, already-correct examples/shader_based_stimuli/
# movingbar/movingbar.py rather than the legacy array-baking moving_bar.py:
# it derives frame count from geometry+speed instead of a hardcoded value,
# and is already a working StimProgram (see GUI plan discussion). Each
# "Generate" writes a small wrapper .py file with the chosen parameters baked
# in, so it shows up as its own selectable, independently playable entry in
# the Instrument tab's file list — the same way each noise file is
# independent — with no changes needed to how the Instrument tab loads .py
# stimuli.

_MOVINGBAR_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "shader_based_stimuli"
    / "movingbar"
    / "movingbar.py"
)

# NOTE: bakes an absolute path to movingbar.py on *this* checkout into the
# generated file. Fine as long as generated stimuli are played from this
# machine/repo; if they need to be portable to another machine, movingbar.py
# (and its .frag shader) would need to be copied alongside instead.
_WRAPPER_TEMPLATE = '''"""Generated moving-bar stimulus config.

Created by fpspy's Generation tab on {timestamp}. Do not hand-edit — use the
Generation tab again to make a new variant instead. Delegates to the real
implementation in movingbar.py so a fix there benefits every generated file.
"""

import importlib.util
from pathlib import Path

_SOURCE = Path({source!r})
_PARAMS = {params!r}


def _load_source():
    spec = importlib.util.spec_from_file_location(_SOURCE.stem, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_program(config):
    return _load_source().MovingBarProgram(**_PARAMS)


def default_config() -> str:
    return ""
'''


def _generate_moving_bar(
    stim_dir: Path,
    file_name: str,
    thickness: float,
    n_directions: int,
    speed: float,
    start_buffer_px: float,
    fps: float,
    n_channels: int,
    width: int,
    height: int,
) -> Path:
    if not _MOVINGBAR_SOURCE.exists():
        raise FileNotFoundError(
            f"movingbar.py not found at {_MOVINGBAR_SOURCE}. Expected it in "
            "this fps_py checkout under "
            "examples/shader_based_stimuli/movingbar/."
        )
    stem = Path(file_name).stem or "moving_bar"
    path = stim_dir / f"{stem}.py"
    params = {
        "stim_shape": (int(height), int(width)),
        "n_channels": int(n_channels),
        "thickness": float(thickness),
        "n_directions": int(n_directions),
        "speed": float(speed),
        "start_buffer_px": float(start_buffer_px),
        "fps": float(fps),
    }
    text = _WRAPPER_TEMPLATE.format(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        source=str(_MOVINGBAR_SOURCE),
        params=params,
    )
    path.write_text(text)
    return path


MOVING_BAR = register(
    Generator(
        key="moving_bar",
        name="Moving bar (shader)",
        params=(
            Param(
                "thickness", "Thickness (px)", "float", 20.0, minimum=1.0, maximum=2000.0
            ),
            Param("n_directions", "Directions", "int", 8, minimum=1, maximum=64),
            Param(
                "speed", "Speed (px/frame)", "float", 2.0, minimum=0.1, maximum=200.0
            ),
            Param(
                "start_buffer_px",
                "Start buffer (px)",
                "float",
                5.0,
                minimum=0.0,
                maximum=500.0,
            ),
            Param("fps", "Frame rate (Hz)", "float", 50.0, minimum=1.0, maximum=240.0),
            Param("n_channels", "LED channels", "int", 6, minimum=1, maximum=8),
            Param("width", "Stimulus width (px)", "int", 500, minimum=1, maximum=10000),
            Param(
                "height", "Stimulus height (px)", "int", 500, minimum=1, maximum=10000
            ),
        ),
        generate=_generate_moving_bar,
    )
)


# --- Random bars ------------------------------------------------------------


def _generate_random_bars(
    stim_dir: Path,
    file_name: str,
    n_bars: int,
    width: int,
    height: int,
    length_min: float,
    length_max: float,
    thickness_min: float,
    thickness_max: float,
    n_orientations: int,
    on_frames: int,
    off_frames: int,
    fps: float,
    seed: int,
) -> Path:
    path = stim_dir / _h5_name(file_name)
    if on_frames < 1:
        raise ValueError(f"A bar must be shown for at least one frame. {on_frames=}")
    # seed=0 means "pick one", but the picked seed is still recorded in the
    # metadata, so any generated file can be regenerated exactly.
    if seed == 0:
        seed = int(np.random.SeedSequence().entropy % (2**63))
    rng = np.random.default_rng(seed)

    bars = fpspy.random_bars.sample_bars(
        rng,
        n_bars=n_bars,
        width=width,
        height=height,
        length_min=length_min,
        length_max=length_max,
        thickness_min=thickness_min,
        thickness_max=thickness_max,
        n_orientations=n_orientations,
    )
    with_blank = off_frames > 0
    frames = fpspy.random_bars.bar_frames(
        bars, width=width, height=height, with_blank=with_blank
    )

    # Two stored frames per bar (the bar, then the blank), each held for its
    # own duration, instead of one stored frame per displayed frame. Durations
    # are given in display frames so that they land on refresh boundaries.
    per_bar = 2 if with_blank else 1
    pattern = [on_frames / fps] + ([off_frames / fps] if with_blank else [])
    frame_durations = np.tile(np.asarray(pattern, dtype=np.float64), n_bars)

    stim_array = fpspy.stim.StimArray(
        frames,
        frame_durations=frame_durations,
        zoom=1,
        # None triggers on every stored frame, i.e. on each bar onset and each
        # blank onset, starting with a bar.
        triggers=None,
        label=(
            f"Random bars, {n_bars} bars, {length_min:g}-{length_max:g}px long, "
            f"{thickness_min:g}-{thickness_max:g}px thick"
        ),
        metadata={
            "n_bars": n_bars,
            "length_min": length_min,
            "length_max": length_max,
            "thickness_min": thickness_min,
            "thickness_max": thickness_max,
            "n_orientations": n_orientations,
            "on_frames": on_frames,
            "off_frames": off_frames,
            "fps": fps,
            "seed": seed,
            "background": 255,
            "bar_value": 0,
        },
    )
    stim_array.write_hdf5(path)
    fpspy.random_bars.append_shape_table(path, bars, per_bar, group="bars")
    return path


RANDOM_BARS = register(
    Generator(
        key="random_bars",
        name="Random bars (flashed, one at a time)",
        params=(
            Param(
                "n_bars",
                "Number of bars",
                "int",
                100,
                minimum=1,
                maximum=100000,
                tooltip="Each bar costs one stored width x height frame (plus "
                "one shared-size blank), so the array is built in memory at "
                "about 1 MB per bar for a 1280x800 field.",
            ),
            Param("width", "Width (px)", "int", 1280, minimum=1, maximum=10000),
            Param("height", "Height (px)", "int", 800, minimum=1, maximum=10000),
            Param(
                "length_min",
                "Bar length min (px)",
                "float",
                100.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "length_max",
                "Bar length max (px)",
                "float",
                800.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "thickness_min",
                "Bar thickness min (px)",
                "float",
                20.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "thickness_max",
                "Bar thickness max (px)",
                "float",
                200.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "n_orientations",
                "Orientations (0 = continuous)",
                "int",
                0,
                minimum=0,
                maximum=360,
                tooltip="0 draws each orientation uniformly from [0, 180). Any "
                "other value picks among that many equally spaced "
                "orientations, e.g. 4 gives 0, 45, 90 and 135 degrees.",
            ),
            Param(
                "on_frames",
                "Bar duration (display frames)",
                "int",
                30,
                minimum=1,
                maximum=100000,
            ),
            Param(
                "off_frames",
                "Blank duration (display frames)",
                "int",
                30,
                minimum=0,
                maximum=100000,
                tooltip="0 leaves no blank between bars: the next bar appears "
                "as the previous one disappears.",
            ),
            Param(
                "fps",
                "Display frame rate (Hz)",
                "float",
                60.0,
                minimum=1.0,
                maximum=240.0,
                tooltip="Only used to turn the two durations above into "
                "seconds. Set it to the rate the stimulus will be played at "
                "so the durations land on refresh boundaries.",
            ),
            Param(
                "seed",
                "Random seed (0 = new each time)",
                "int",
                0,
                minimum=0,
                maximum=2**31 - 1,
                tooltip="The seed actually used is always written to the "
                "file's metadata, so any generated file can be reproduced.",
            ),
        ),
        generate=_generate_random_bars,
    )
)


# --- Random dots ------------------------------------------------------------
# Same stimulus as the random bars, with a filled circle in place of the
# rotated rectangle: random centre, random diameter, one at a time, each
# followed by a blank. A dot has no orientation, so diameter is the only shape
# parameter.


def _generate_random_dots(
    stim_dir: Path,
    file_name: str,
    n_dots: int,
    width: int,
    height: int,
    diameter_min: float,
    diameter_max: float,
    on_frames: int,
    off_frames: int,
    fps: float,
    seed: int,
) -> Path:
    path = stim_dir / _h5_name(file_name)
    if on_frames < 1:
        raise ValueError(f"A dot must be shown for at least one frame. {on_frames=}")
    # seed=0 means "pick one", but the picked seed is still recorded in the
    # metadata, so any generated file can be regenerated exactly.
    if seed == 0:
        seed = int(np.random.SeedSequence().entropy % (2**63))
    rng = np.random.default_rng(seed)

    dots = fpspy.random_dots.sample_dots(
        rng,
        n_dots=n_dots,
        width=width,
        height=height,
        diameter_min=diameter_min,
        diameter_max=diameter_max,
    )
    with_blank = off_frames > 0
    frames = fpspy.random_dots.dot_frames(
        dots, width=width, height=height, with_blank=with_blank
    )

    # Two stored frames per dot (the dot, then the blank), each held for its
    # own duration, instead of one stored frame per displayed frame. Durations
    # are given in display frames so that they land on refresh boundaries.
    per_dot = 2 if with_blank else 1
    pattern = [on_frames / fps] + ([off_frames / fps] if with_blank else [])
    frame_durations = np.tile(np.asarray(pattern, dtype=np.float64), n_dots)

    stim_array = fpspy.stim.StimArray(
        frames,
        frame_durations=frame_durations,
        zoom=1,
        # None triggers on every stored frame, i.e. on each dot onset and each
        # blank onset, starting with a dot.
        triggers=None,
        label=(
            f"Random dots, {n_dots} dots, "
            f"{diameter_min:g}-{diameter_max:g}px across"
        ),
        metadata={
            "n_dots": n_dots,
            "diameter_min": diameter_min,
            "diameter_max": diameter_max,
            "on_frames": on_frames,
            "off_frames": off_frames,
            "fps": fps,
            "seed": seed,
            "background": 255,
            "dot_value": 0,
        },
    )
    stim_array.write_hdf5(path)
    fpspy.random_bars.append_shape_table(path, dots, per_dot, group="dots")
    return path


RANDOM_DOTS = register(
    Generator(
        key="random_dots",
        name="Random dots (flashed, one at a time)",
        params=(
            Param(
                "n_dots",
                "Number of dots",
                "int",
                100,
                minimum=1,
                maximum=100000,
                tooltip="Each dot costs one stored width x height frame (plus "
                "one shared-size blank), so the array is built in memory at "
                "about 1 MB per dot for a 1280x800 field.",
            ),
            Param("width", "Width (px)", "int", 1280, minimum=1, maximum=10000),
            Param("height", "Height (px)", "int", 800, minimum=1, maximum=10000),
            Param(
                "diameter_min",
                "Dot diameter min (px)",
                "float",
                20.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "diameter_max",
                "Dot diameter max (px)",
                "float",
                400.0,
                minimum=1.0,
                maximum=10000.0,
            ),
            Param(
                "on_frames",
                "Dot duration (display frames)",
                "int",
                30,
                minimum=1,
                maximum=100000,
            ),
            Param(
                "off_frames",
                "Blank duration (display frames)",
                "int",
                30,
                minimum=0,
                maximum=100000,
                tooltip="0 leaves no blank between dots: the next dot appears "
                "as the previous one disappears.",
            ),
            Param(
                "fps",
                "Display frame rate (Hz)",
                "float",
                60.0,
                minimum=1.0,
                maximum=240.0,
                tooltip="Only used to turn the two durations above into "
                "seconds. Set it to the rate the stimulus will be played at "
                "so the durations land on refresh boundaries.",
            ),
            Param(
                "seed",
                "Random seed (0 = new each time)",
                "int",
                0,
                minimum=0,
                maximum=2**31 - 1,
                tooltip="The seed actually used is always written to the "
                "file's metadata, so any generated file can be reproduced.",
            ),
        ),
        generate=_generate_random_dots,
    )
)
