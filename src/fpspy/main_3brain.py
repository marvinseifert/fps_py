"""
This is the main file for the project. It starts two processes, one for the GUI and one
for the stimulus presentation. The GUI is implemented using tkinter and the stimulus
presentation is implemented using pyglet.

Author: Marvin Seifert
"""

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)

import multiprocessing as mp
import typer
from pathlib import Path
import logging
import time
import fpspy.config
import fpspy.gui
import fpspy.play_3brain
import fpspy.stim
import fpspy.queue


gui_app = typer.Typer(help="fpspy GUI. Preset visual stimuli with OpenGL.")
cli_app = typer.Typer(help="fpspy CLI. Preset visual stimuli with OpenGL.")

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5679, stdout_to_server=True, stderr_to_server=True, suspend=False)


@gui_app.command()
def run_gui(
    config_path: Path | None = typer.Argument(
        None,
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        case_sensitive=False,
    ),
):
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = fpspy.config.load_config(config_path)
    n_windows = len(config["windows"])

    # Delay between loading the stimulus and the start of the presentation, in seconds.
    presentation_delay = 10

    # Start the GUI and the stimulus presentation in separate processes.
    cmd_queue = mp.Queue()  
    status_queue = mp.Queue()
    # Gui process
    p1 = mp.Process(
        target=fpspy.gui.tkinter_app,
        args=(
            config,
            cmd_queue,
            status_queue,
            n_windows,
        ),
    )
    # Presentation lead process
    p2 = mp.Process(
        target=fpspy.play_3brain.pyglet_app_lead,
        args=(
            1,
            config,
            cmd_queue,
            status_queue,
            presentation_delay,
        ),
    )  # Start the pyglet app
    # Start the processes
    p1.start()
    p2.start()

    # Presentation follow processes
    follow_processes = []
    for idx in range(2, n_windows + 1):
        p = mp.Process(
            target=fpspy.play_3brain.pyglet_app_follow,
            args=(
                idx,
                config,
                cmd_queue,
                status_queue,
                presentation_delay,
            ),
        )
        p.start()
        follow_processes.append(p)

    # p4.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()
    for p in follow_processes:
        p.join()
    # p4.join()


@cli_app.command()
def run_cli(
    stim_path: Path = typer.Argument(
        ...,
        help="Path to stimulus file (.h5)",
    ),
    config_path: Path | None = typer.Option(
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
    arduino_colours: str = typer.Option(
        "R,G,B,U",
        "--colours",
        help="Arduino colour logic (e.g., R,G,B,U)",
    ),
    change_every: int = typer.Option(
        100,
        "--change-every",
        help="Change colour logic every N frames",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        case_sensitive=False,
    ),
):
    """Run a stimulus from the command line without the GUI."""
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = fpspy.config.load_config(config_path)
    n_windows = len(config["windows"])
    # Validate stimulus path
    if not stim_path.exists():
        typer.echo(f"Error: stimulus file not found: {stim_path}", err=True)
        raise typer.Exit(1)

    # Load stimulus info
    try:
        info = fpspy.stim.Stim.preview_hdf5(stim_path)
    except Exception as e:
        typer.echo(f"Error loading stimulus file: {e}", err=True)
        raise typer.Exit(1)

    # Schedule frames
    def schedule_frames(frames, frame_rate):
        current_time = time.perf_counter()
        fps = frame_rate
        frame_duration = 1 / fps
        s_frames = __import__('numpy').linspace(
            current_time, current_time + frames * frame_duration, frames + 1
        )
        return s_frames

    s_frames = schedule_frames(info["n_frames"], info["fps"])

    # Delay between loading the stimulus and the start of the presentation, in seconds.
    presentation_delay = 10

    # Create queues for inter-process communication
    cmd_queue = mp.Queue()
    status_queue = mp.Queue()

    # Put play command in queue for all windows
    for _ in range(n_windows):
        fpspy.queue.put(
            cmd_queue,
            "play",
            stim_path=stim_path,
            loops=loops,
            arduino_colours=arduino_colours,
            change_logic=change_every,
            s_frames=s_frames,
        )

    typer.echo(f"Playing stimulus: {stim_path}")
    typer.echo(f"  Frames: {info['n_frames']}, FPS: {info['fps']}, Loops: {loops}")

    # Start presentation lead process
    p_lead = mp.Process(
        target=fpspy.play_3brain.pyglet_app_lead,
        args=(
            1,
            config,
            cmd_queue,
            status_queue,
            presentation_delay,
        ),
    )
    p_lead.start()

    # Start presentation follow processes
    follow_processes = []
    for idx in range(2, n_windows + 1):
        p = mp.Process(
            target=fpspy.play_3brain.pyglet_app_follow,
            args=(
                idx,
                config,
                cmd_queue,
                status_queue,
                presentation_delay,
            ),
        )
        p.start()
        follow_processes.append(p)

    # Wait for processes to finish
    p_lead.join()
    for p in follow_processes:
        p.join()

    typer.echo("Stimulus playback completed.")


def gui():
    gui_app()

def cli():
    cli_app()


if __name__ == "__main__":
    gui()
