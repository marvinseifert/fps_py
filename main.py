# Description: This is the main file for the project. It starts two processes, one for the GUI and one for the noise
# presentation. The GUI is implemented using tkinter and the noise presentation is implemented using pyglet.
# Author: Marvin Seifert
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)

from multiprocessing import Process, Queue, Lock
from main_gui import tkinter_app
from play_noise import pyglet_app_lead, pyglet_app_follow
import window_settings
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5679, stdout_to_server=True, stderr_to_server=True, suspend=False)

# Load the window settings
windows = window_settings.get_windows()

# Configuration dictionary for the pyglet app window. Change according to your needs.
config_dict = {"windows": windows, "gl_version": (4, 1), "fps": 75}

nr_windows = len(windows)

presentation_delay = 10  # Delay between loading of the stimulus to the start of the presentation in seconds


# Start the GUI and the noise presentation in separate processes
if __name__ == "__main__":
    queue1 = Queue()  # Queue for communication between all processes
    sync_queue = Queue()  # Queue for synchronization between the presentation processes
    sync_lock = Lock()
    queue_lock = Lock()
    arduino_queue = Queue()
    arduino_lock = Lock()
    status_queue = Queue()
    status_lock = Lock()
    # Gui process
    p1 = Process(
        target=tkinter_app,
        args=(
            queue1,
            queue_lock,
            arduino_queue,
            arduino_lock,
            status_queue,
            status_lock,
            nr_windows,
        ),
    )
    # Presentation lead process
    p2 = Process(
        target=pyglet_app_lead,
        args=(
            1,
            config_dict,
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
    for idx in range(2, nr_windows + 1):
        p = Process(
            target=pyglet_app_follow,
            args=(
                idx,
                config_dict,
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
