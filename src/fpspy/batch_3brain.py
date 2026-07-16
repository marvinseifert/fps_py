"""Play a sequence of stimuli (a "playlist") using one set of presenter windows.

The presenter processes are started once; each stimulus is played back-to-back,
with windows showing their clear color in between. If any presenter process
fails, the playlist stops and raises.

Requires presenters to report completion on the status queue: the
{"play_finished": ...} message put in Presenter.communicate after a play.

Example (the __main__ guard is required on Windows, where multiprocessing
spawns fresh interpreters):

    import fpspy.batch_3brain as batch 

    if __name__ == "__main__":
        batch.play_playlist(
            [
                batch.Item(
                    "examples/shader_based_stimuli/movingbar/movingbar.py",
                    stim_config_path=(
                        "examples/shader_based_stimuli/movingbar/3led_10tk_2s.json"
                    ),
                    label="movingbar_3led",
                ),
                batch.Item(
                    "../data/stim/gaussian_checkerboard_768l-8s-10Hz-5min.h5",
                    lazy_textures=True,
                ),
            ],
            config_path="../_configs/fpspy.toml",
            enable_triggers=False,
        )

Giving an Item a `label` puts "S:<label>" on the trigger wire before the
stimulus and "E:<label>" after it, so the recording can be segmented offline
without a separate clock. This needs enable_triggers and a connected Arduino;
labels are ignored otherwise. Each marker costs (8 + len) * 10 ms of wire time,
during which the Arduino cannot send triggers, so the playlist waits for each
one to finish before starting the stimulus. Keep labels short.
"""

import dataclasses
import json
import logging
import queue as std_queue
import time
from pathlib import Path
from typing import Iterable, List, Optional, Union

import fpspy.config
import fpspy.play_3brain
import fpspy.queue
import fpspy.stim

_logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# Labels are put on the trigger wire either side of a stimulus, so a recording
# can be segmented offline without a separate clock. The prefixes pair up: an
# "E:" is unambiguous even if noise costs you the matching "S:".
START_PREFIX = "S:"
END_PREFIX = "E:"


def start_msg(label: str) -> str:
    """Text to put on the trigger wire before a stimulus."""
    return f"{START_PREFIX}{label}"


def end_msg(label: str) -> str:
    """Text to put on the trigger wire after a stimulus."""
    return f"{END_PREFIX}{label}"


class Item:
    """One entry of a playlist."""

    stim_path: PathLike
    # Config file for the stimulus program. Ignored for .h5 stimuli.
    stim_config_path: Optional[PathLike] = None
    loops: int = 1
    # Only applies to .h5 (TextureSequence) stimuli.
    lazy_textures: bool = False
    # Sent on the trigger wire as "S:<label>" before the stimulus and
    # "E:<label>" after it, to mark the stimulus out in the recording. Needs
    # enable_triggers and a connected Arduino; ignored otherwise.
    label: Optional[str] = None

    def __init__(
        self,
        stim_path: PathLike,
        stim_config_path: Optional[PathLike] = None,
        loops: int = 1,
        lazy_textures: bool = False,
        label: Optional[str] = None,
    ):
        self.stim_path = stim_path
        self.stim_config_path = stim_config_path
        self.loops = loops
        self.lazy_textures = lazy_textures
        if label is None:
            label = Path(stim_path).name
        self.label = label

    def stim_config(self) -> Optional[str]:
        """Build the stim_config string, mirroring the CLI's play command."""
        if Path(self.stim_path).suffix.lower() == ".h5":
            return json.dumps({"lazy_textures": self.lazy_textures})
        if self.stim_config_path is not None:
            return Path(self.stim_config_path).read_text(encoding="utf-8")
        return None


@dataclasses.dataclass
class ItemInfo:
    """Timing information for one playlist item."""

    stim_path: Path
    label: Optional[str]
    loops: int
    # Seconds for all loops, or None if the stimulus can't report its duration
    # without opening a window (e.g. movies).
    duration: Optional[float]


@dataclasses.dataclass
class PlaylistInfo:
    """Timing information for a playlist. Returned by info()."""

    items: List[ItemInfo]
    # Inter-stimulus delay applied before every item.
    delay: float

    def total_duration(self) -> Optional[float]:
        """Total running time in seconds, or None if any item is unknown."""
        durations = [item.duration for item in self.items]
        if any(d is None for d in durations):
            return None
        return sum(durations) + self.delay * len(self.items)

    def __str__(self) -> str:
        lines = []
        for i, item in enumerate(self.items):
            name = (
                item.label
                if item.label is not None
                else Path(item.stim_path).name
            )
            loops_str = f" x{item.loops}" if item.loops != 1 else ""
            lines.append(
                f"[{i + 1}/{len(self.items)}] {_fmt_duration(item.duration):>8} "
                f" {name}{loops_str}"
            )
        lines.append(
            f"Total: {_fmt_duration(self.total_duration())}"
            + (
                f" (includes {self.delay:g} s delay per item)"
                if self.delay
                else ""
            )
        )
        return "\n".join(lines)


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _item_duration(item: Item) -> Optional[float]:
    """Running time of one item in seconds (all loops), without opening a window.

    Known for .h5 stimuli (from the file's frame durations, including any
    channel-mask repeats) and for script programs that expose `n_frames` and
    `fps` attributes. None otherwise (e.g. movies, which are only probed
    during setup()).
    """
    prog = fpspy.stim.create_program(Path(item.stim_path), item.stim_config())
    duration = None
    if isinstance(prog, fpspy.stim.TextureSequence):
        duration = float(prog.stim_arr.frame_times()[-1])
    else:
        n_frames = getattr(prog, "n_frames", None)
        fps = getattr(prog, "fps", None)
        if (
            isinstance(n_frames, int)
            and isinstance(fps, (int, float))
            and fps > 0
        ):
            duration = n_frames / fps
    if duration is not None:
        duration *= item.loops
    return duration


def info(items: Iterable[Item], delay: float = 0.0) -> PlaylistInfo:
    """Report the running time of each item and of the whole playlist.

    `delay` is the inter-stimulus gap that play_playlist will apply before
    every item (the config's presentation_delay); pass it to have the total
    account for the gaps. Durations don't include trigger-wire marker time.

    print() the result for a readable summary, or use the fields directly.
    """
    item_infos = [
        ItemInfo(
            stim_path=Path(item.stim_path),
            label=item.label,
            loops=item.loops,
            duration=_item_duration(item),
        )
        for item in items
    ]
    return PlaylistInfo(items=item_infos, delay=delay)


def play_playlist(
    items: Iterable[Item],
    config_path: Optional[PathLike] = None,
    delay: Optional[float] = None,
    out_dir: Optional[PathLike] = None,
    enable_triggers: bool = True,
    log_level: str = "INFO",
    label: Optional[str] = None,
) -> None:
    """Play each item in sequence, reusing one set of presenter windows.

    `delay` (default: the config's presentation_delay) applies before every
    item, acting as the inter-stimulus gap. Raises RuntimeError if a presenter
    process dies; remaining items are not played.

    `label` marks the playlist as a whole: it goes on the trigger wire as
    "S:<label>" before any stimulus starts and "E:<label>" after the last one
    finishes. Like Item labels, it needs enable_triggers and a connected
    Arduino, and is ignored otherwise.
    """
    items = list(items)
    if not items:
        raise ValueError("Empty playlist.")
    # Validate everything up front, before any window opens.
    if label is not None:
        _check_label(label)
    for item in items:
        if not Path(item.stim_path).exists():
            raise FileNotFoundError(
                f"Stimulus file not found: {item.stim_path}"
            )
        if (
            item.stim_config_path is not None
            and not Path(item.stim_config_path).exists()
        ):
            raise FileNotFoundError(
                f"Stimulus config not found: {item.stim_config_path}"
            )
        if item.label is not None:
            _check_label(item.label)

    config = fpspy.config.load_config(
        Path(config_path) if config_path is not None else None
    )
    if delay is None:
        delay = fpspy.config.get_presentation_delay(config)
    if out_dir is None:
        out_dir = fpspy.config.create_outdir(config)
    _logger.info(f"{out_dir} [output dir]")

    processes, cmd_queues, status_queue = (
        fpspy.play_3brain.start_presenter_processes(
            config, out_dir, delay, enable_triggers, log_level
        )
    )
    completed_ok = False
    try:
        if enable_triggers and label is not None:
            _send_marker(cmd_queues, status_queue, processes, start_msg(label))
        for i, item in enumerate(items):
            _logger.info(f"[{i + 1}/{len(items)}] {item.stim_path}")
            stim_config = item.stim_config()
            if enable_triggers and item.label is not None:
                _send_marker(
                    cmd_queues, status_queue, processes, start_msg(item.label)
                )
            # A fresh reference time per item: frame schedules are t0-relative
            # and stim.delay() raises if the schedule start is in the past.
            t0 = time.perf_counter()
            for q in cmd_queues:
                fpspy.queue.put_onto(
                    q,
                    "play",
                    stim_path=Path(item.stim_path),
                    stim_config=stim_config,
                    loops=item.loops,
                    t0=t0,
                    close_after=False,
                )
            _wait_for_play_finished(status_queue, processes, item)
            if enable_triggers and item.label is not None:
                _send_marker(
                    cmd_queues, status_queue, processes, end_msg(item.label)
                )
            _logger.info(f"[{i + 1}/{len(items)}] finished")
        if enable_triggers and label is not None:
            _send_marker(cmd_queues, status_queue, processes, end_msg(label))
        completed_ok = True
    finally:
        if completed_ok:
            for q in cmd_queues:
                fpspy.queue.put_onto(q, "destroy")
        else:
            # A presenter died or the coordinator raised: don't leave the
            # remaining presenter windows running.
            for p in processes:
                if p.is_alive():
                    p.terminate()
        for p in processes:
            p.join()
    _logger.info("Playlist completed.")


def _check_label(label: str) -> None:
    """Reject labels the trigger wire cannot carry, before any window opens."""
    if "\n" in label:
        raise ValueError(
            f"Label cannot contain a newline (it terminates the message): {label!r}"
        )
    # Both markers must fit, and the limit is bytes of UTF-8, not characters.
    for prefix in (START_PREFIX, END_PREFIX):
        max_text_bytes = 255  # see arduino.py
        n = len(f"{prefix}{label}".encode("utf-8"))
        if n > max_text_bytes:
            raise ValueError(
                f"Label is too long: {prefix!r} + label is {n} bytes of UTF-8, "
                f"max is {max_text_bytes}: {label!r}"
            )


def _send_marker(cmd_queues, status_queue, processes, text: str) -> None:
    """Put `text` on the trigger wire and block until the Arduino is done.

    Only the first presenter owns the Arduino (see start_presenter_processes),
    so only it is asked. Waiting matters: the Arduino cannot service triggers
    while transmitting, so a stimulus must not start until this returns.
    """
    _logger.info(f"{text} [trigger wire]")
    fpspy.queue.put_onto(cmd_queues[0], "message", text)
    _wait_for_status(
        status_queue, processes, "message_sent", 1, f"sending {text!r}"
    )


def _wait_for_status(
    status_queue, processes, key: str, n: int, what: str
) -> None:
    """Block until `n` presenters report `key` on the status queue."""
    seen = 0
    while seen < n:
        try:
            msg = status_queue.get(timeout=1.0)
        except std_queue.Empty:
            dead = [p for p in processes if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    f"Presenter process died (exit code {dead[0].exitcode}) "
                    f"while {what}. Stopping playlist."
                )
            continue
        if isinstance(msg, dict) and key in msg:
            seen += 1
        # Other status messages (e.g. "done") are ignored.


def _wait_for_play_finished(status_queue, processes, item) -> None:
    """Block until every presenter reports finishing the current item."""
    _wait_for_status(
        status_queue,
        processes,
        "play_finished",
        len(processes),
        f"playing {item.stim_path}",
    )
