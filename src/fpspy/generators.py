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