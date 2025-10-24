"""
This file contains the settings for the windows used in the application. Each window is defined by a dictionary with various parameters.
Just look at the example and change the values according to your needs.
@ Marvin Seifert, 2025
"""
import numpy as np


def get_windows():
    """
    Returns the window settings for the application.
    Each window is defined by a dictionary with various parameters.
    """
    windows = {
        "1": {
            "y_shift": 1000,
            "x_shift": -1080,
            "window_size": (500, 500),
            "fullscreen": False,
            "style": "transparent",
            "channels": np.array([0, 1, 2]),
            "arduino_port": "/dev/ttyUSB0",
            "arduino_baud_rate": 9600,
        }
    }

    # Add more windows as needed

    #     "2": {
    #         "y_shift": 0,
    #         "x_shift": -1000,
    #         "window_size": (900, 900),
    #         "fullscreen": False,
    #         "style": "transparent",
    #         "channels": np.array([0, 1, 2]),
    #         "arduino_port": "COM3",
    #         "arduino_baud_rate": 9600,
    #     },
    # }
    return windows
