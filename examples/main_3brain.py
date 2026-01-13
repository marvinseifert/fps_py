"""Main entry point for our use of fpspy with our 3brain setup."""

import multiprocessing as mp
import logging
import typer
from pathlib import Path
import time
from typing import Literal, Optional
import numpy as np
import fpspy.config
import fpspy.gui
import fpspy.play_3brain
import fpspy.stim
import fpspy.queue
import fpspy._logging as _logging

_logger = logging.getLogger(__name__)


StimType = Literal["auto", "h5", "shader"]


gui_app = typer.Typer(help="fpspy GUI. Preset visual stimuli with OpenGL.")
cli_app = typer.Typer(help="fpspy CLI. Preset visual stimuli with OpenGL.")


def _preview_h5_stim(stim_path: Path, loops: int):
    """Load and preview HDF5 stimulus, logging info."""
    info = fpspy.stim.StimArray.preview_hdf5(stim_path)
    _logger.info(
        f"Playing HDF5 stimulus (loops={loops}): "
        f"{stim_path} (version: {info['version']})"
    )
    _logger.info(info)
    return info


@cli_app.command()
def play(
    stim_path: Path = typer.Argument(
        ...,
        help="Path to stimulus file (.h5 for precomputed, .json for shader-based)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    loops: int = typer.Option(
        1,
        "--loops",
        "-l",
        help="Number of times to loop the stimulus",
    ),
    delay: Optional[float] = typer.Option(
        None,
        "--delay",
        "-d",
        help="Delay before starting the stimulus presentation (in seconds)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help=(
            "Directory to save logs and output data. If omitted, use "
            f"{fpspy.config.default_log_dir()}/<timestamp>/."
        ),
    ),
    enable_triggers: bool = typer.Option(
        True,
        "--triggers/--no-triggers",
        help="Enable Arduino triggers during presentation.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
):
    """Run a stimulus from the command line without the GUI."""
    log_level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    _logging.setup_main_logging(log_level)

    # Validate stimulus file exists.
    if not stim_path.exists():
        _logger.error(f"Error: stimulus file not found: {stim_path}")
        raise typer.Exit(1)

    # Load configuration.
    config = fpspy.config.load_config(config_path)
    if delay is None:
        delay = fpspy.config.get_presentation_delay(config)
    if out_dir is None:
        out_dir = fpspy.config.create_outdir(config)

    _logger.info(f"{out_dir} [output dir]")

    if stim_path.suffix.lower() == ".h5":
        _preview_h5_stim(stim_path, loops)

    # Create a reference time point.
    t0 = time.perf_counter()

    # Start presenter processes, and wait to finish.
    presenter_processes, cmd_queues, status_queue = (
        fpspy.play_3brain.start_presenter_processes(
            config, out_dir, delay, enable_triggers, log_level
        )
    )
    play_cmd = "play"
    # Create queues for inter-process communication.
    for queue in cmd_queues:
        fpspy.queue.put_onto(
            queue,
            play_cmd,
            stim_path=stim_path,
            stim_config=None,
            loops=loops,
            t0=t0,
            close_after=True,
        )

    for p in presenter_processes:
        p.join()

    _logger.info("Stimulus playback completed.")


@cli_app.command()
def export(
    stim_path: Path = typer.Argument(
        ...,
        help="Path to stimulus file (.py for shader-based)",
    ),
    stim_config_path: Optional[Path] = typer.Option(
        None,
        "--stim-config",
        "-s",
        help="Optional config file (arbitrary text file) for the stimulus program.",
    ),
    out_path: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help=(
            "Path to save the exported stimulus .npy file. If omitted, use "
            "<stimulus_filename>_exported.h5"
        ),
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
):
    """Render a stimulus to numpy arrays using offscreen GPU rendering."""
    log_level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    _logging.setup_main_logging(log_level)

    if out_path is None:
        out_path = stim_path.with_name(stim_path.stem + "_exported.h5")
    _logger.info(f"Exporting stimulus as fpspy.stim.ArrayStim to: {out_path}")
    if not stim_path.exists():
        _logger.error(f"Error: stimulus file not found: {stim_path}")
        raise typer.Exit(1)

    # Load configuration.
    config = fpspy.config.load_config(config_path)
    if stim_config_path is not None:
        with stim_config_path.open("r", encoding="utf-8") as f:
            stim_config = f.read()
    else:
        stim_config = None
    prog = fpspy.stim.create_program(stim_path, stim_config)
    stim = fpspy.play_3brain.export(prog, config)
    stim.write_hdf5(out_path)


@gui_app.command()
def gui(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        "-o",
        help=(
            "Directory to save logs and output data. If omitted, use "
            f"{fpspy.config.default_log_dir()}."
        ),
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
):
    log_level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    _logging.setup_main_logging(log_level)

    config = fpspy.config.load_config(config_path)
    arduino_queue = mp.Queue()  # Dummy queue for 3brain (not used)
    delay = fpspy.config.get_presentation_delay(config)
    out_dir = fpspy.config.create_outdir(config)

    # Start presenter processes.
    presenter_processes, cmd_queues, status_queue = (
        fpspy.play_3brain.start_presenter_processes(
            config, out_dir, delay, True, log_level
        )
    )

    # Start GUI process.
    n_windows = len(config["windows"])
    gui_process = mp.Process(
        target=fpspy.gui.tkinter_app,
        args=(config, cmd_queues, arduino_queue, status_queue, n_windows),
    )
    gui_process.start()

    # Wait for all processes to finish.
    for p in [gui_process] + presenter_processes:
        p.join()


def run_gui():
    gui_app()


def run_cli():
    cli_app()


if __name__ == "__main__":
    run_gui()
