"""
This is the main file for the project. It starts two processes, one for the GUI and one
for the noise presentation. The GUI is implemented using tkinter and the noise
presentation is implemented using pyglet.

Author: Marvin Seifert
"""

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)

from multiprocessing import Process, Queue, Lock
import typer
from pathlib import Path
import logging
import fpspy.config
import fpspy.main_gui
import fpspy.play


app = typer.Typer(help="fpspy—preset visual stimuli with OpenGL.")

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5679, stdout_to_server=True, stderr_to_server=True, suspend=False)


@app.command()
def run(
    config_path: Path | None = typer.Argument(
        None,
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = fpspy.config.load_config(config_path)
    n_windows = len(config["windows"])

    # Delay between loading the stimulus and the start of the presentation, in seconds.
    presentation_delay = 10  

    # Start the GUI and the noise presentation in separate processes.
    queue1 = Queue()  # Queue for communication between all processes.
    sync_queue = (
        Queue()
    )  # Queue for synchronization between the presentation processes.
    sync_lock = Lock()
    queue_lock = Lock()
    arduino_queue = Queue()
    arduino_lock = Lock()
    status_queue = Queue()
    status_lock = Lock()
    # Gui process
    p1 = Process(
        target=fpspy.main_gui.tkinter_app,
        args=(
            config,
            queue1,
            queue_lock,
            arduino_queue,
            arduino_lock,
            status_queue,
            status_lock,
            n_windows,
        ),
    )
    # Presentation lead process
    p2 = Process(
        target=fpspy.play.pyglet_app_lead,
        args=(
            1,
            config,
            queue1,
            sync_queue,
            sync_lock,
            queue_lock,
            arduino_queue,
            arduino_lock,
            status_queue,
            status_lock,
            presentation_delay,
        ),
    )  # Start the pyglet app
    # Start the processes
    p1.start()
    p2.start()

    # Presentation follow processes
    follow_processes = []
    for idx in range(2, n_windows + 1):
        p = Process(
            target=fpspy.play.pyglet_app_follow,
            args=(
                idx,
                config,
                queue1,
                sync_queue,
                sync_lock,
                queue_lock,
                arduino_queue,
                arduino_lock,
                status_queue,
                status_lock,
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


def cli_main():
    app()

def gui_main():
    run(config_path=None)
