# Description: This is the main file for the project. It starts two processes, one for the GUI and one for the noise
# presentation. The GUI is implemented using tkinter and the noise presentation is implemented using pyglet.
# Author: Marvin Seifert

from multiprocessing import Process, Queue, Lock
from main_gui import tkinter_app
from play_noise import pyglet_app_lead, pyglet_app_follow
from arduino import run_arduino
from multiprocessing import shared_memory
import numpy as np

windows = {
    "1": {
        "y_shift": -100,
        "x_shift": 2560,
        "window_size": (800, 800),
        "fullscreen": False,
        "style": "borderless",
    },
    "2": {
        "y_shift": 200,
        "x_shift": 2600,
        "window_size": (800, 800),
        "fullscreen": False,
        "style": "borderless",
    },
    "3": {
        "y_shift": 600,
        "x_shift": 2600,
        "window_size": (200, 200),
        "fullscreen": False,
        "style": "borderless",
    },
    # "4": {
    #     "y_shift": 800,
    #     "x_shift": 2600,
    #     "window_size": (200, 200),
    #     "fullscreen": False,
    # },
}


# Configuration dictionary for the pyglet app window. Change according to your needs.
config_dict = {"windows": windows, "gl_version": (4, 1), "fps": 60}

nr_windows = len(windows)

arduino_port = "COM3"
arduino_baud_rate = 9600


# Start the GUI and the noise presentation in separate processes
if __name__ == "__main__":
    queue1 = Queue()  # Queue for communication between all processes
    sync_queue = Queue()  # Queue for synchronization between the presentation processes
    sync_lock = Lock()
    queue_lock = Lock()
    arduino_queue = Queue()
    arduino_lock = Lock()
    # Gui process
    p1 = Process(
        target=tkinter_app,
        args=(queue1, queue_lock, arduino_queue, arduino_lock, nr_windows),
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
        ),
    )  # Start the pyglet app
    # Arduino process
    pA = Process(
        target=run_arduino,
        args=(arduino_port, arduino_baud_rate, arduino_queue, arduino_lock),
    )
    # Start the processes
    p1.start()
    p2.start()
    pA.start()
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
            ),
        )
        p.start()
        follow_processes.append(p)

    # p4.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()
    pA.join()
    for p in follow_processes:
        p.join()
    # p4.join()
