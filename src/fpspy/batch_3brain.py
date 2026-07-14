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
                ),
                batch.Item(
                    "../data/stim/gaussian_checkerboard_768l-8s-10Hz-5min.h5",
                    lazy_textures=True,
                ),
            ],
            config_path="../_configs/fpspy.toml",
            enable_triggers=False,
        )
"""
import dataclasses
import json
import logging
import queue as std_queue
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import fpspy.config
import fpspy.play_3brain
import fpspy.queue

_logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclasses.dataclass
class Item:
    """One entry of a playlist."""

    stim_path: PathLike
    # Config file for the stimulus program. Ignored for .h5 stimuli.
    stim_config_path: Optional[PathLike] = None
    loops: int = 1
    # Only applies to .h5 (TextureSequence) stimuli.
    lazy_textures: bool = False

    def stim_config(self) -> Optional[str]:
        """Build the stim_config string, mirroring the CLI's play command."""
        if Path(self.stim_path).suffix.lower() == ".h5":
            return json.dumps({"lazy_textures": self.lazy_textures})
        if self.stim_config_path is not None:
            return Path(self.stim_config_path).read_text(encoding="utf-8")
        return None


def play_playlist(
    items: Iterable[Item],
    config_path: Optional[PathLike] = None,
    delay: Optional[float] = None,
    out_dir: Optional[PathLike] = None,
    enable_triggers: bool = True,
    log_level: str = "INFO",
) -> None:
    """Play each item in sequence, reusing one set of presenter windows.

    `delay` (default: the config's presentation_delay) applies before every
    item, acting as the inter-stimulus gap. Raises RuntimeError if a presenter
    process dies; remaining items are not played.
    """
    items = list(items)
    if not items:
        raise ValueError("Empty playlist.")
    # Validate everything up front, before any window opens.
    for item in items:
        if not Path(item.stim_path).exists():
            raise FileNotFoundError(f"Stimulus file not found: {item.stim_path}")
        if (
            item.stim_config_path is not None
            and not Path(item.stim_config_path).exists()
        ):
            raise FileNotFoundError(
                f"Stimulus config not found: {item.stim_config_path}"
            )

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
        for i, item in enumerate(items):
            _logger.info(f"[{i + 1}/{len(items)}] {item.stim_path}")
            stim_config = item.stim_config()
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
            _logger.info(f"[{i + 1}/{len(items)}] finished")
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


def _wait_for_play_finished(status_queue, processes, item) -> None:
    """Block until every presenter reports finishing the current item."""
    n_finished = 0
    while n_finished < len(processes):
        try:
            msg = status_queue.get(timeout=1.0)
        except std_queue.Empty:
            dead = [p for p in processes if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    f"Presenter process died (exit code {dead[0].exitcode}) "
                    f"while playing {item.stim_path}. Stopping playlist."
                )
            continue
        if isinstance(msg, dict) and "play_finished" in msg:
            n_finished += 1
        # Other status messages (e.g. "done") are ignored.
