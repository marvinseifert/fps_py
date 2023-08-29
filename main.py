# Description: This is the main file for the project. It starts two processes, one for the GUI and one for the noise
# presentation. The GUI is implemented using tkinter and the noise presentation is implemented using pyglet.
# Author: Marvin Seifert

from multiprocessing import Process, Queue
from main_gui import tkinter_app
from play_noise import pyglet_app



# Configuration dictionary for the pyglet app window. Change according to your needs.
config_dict = {
    "y_shift": -500,
    "x_shift": 2560,
    "gl_version": (4, 1),
    "window_size": (1080, 1920),
    "fullscreen": False}


# Start the GUI and the noise presentation in separate processes
if __name__ == '__main__':
    queue = Queue() # Queue for communication between the processes

    p1 = Process(target=tkinter_app, args=(queue,)) # Start the GUI
    p2 = Process(target=pyglet_app, args=(config_dict, queue)) # Start the pyglet app

    p1.start()
    p2.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()
