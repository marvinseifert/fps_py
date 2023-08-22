from multiprocessing import Process, Queue
from main_gui import tkinter_app
from play_noise import pyglet_app


config_dict = {
    "y_shift": 2560,
    "x_shift": -500,
    "gl_version": (4, 1),
    "window_size": (1080, 1920),
    "fullscreen": False}



if __name__ == '__main__':
    queue = Queue()

    p1 = Process(target=tkinter_app, args=(queue,))
    p2 = Process(target=pyglet_app, args=(config_dict, queue))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
