import moderngl_window
import pyglet
import moderngl
from moderngl_window.conf import settings
import time
from multiprocessing import Process, Queue
import h5py
import numpy as np
from pathlib import Path

class Presenter:

    def __init__(self, config_dict, queue):
        self.queue = queue
        settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'
        settings.WINDOW['gl_version'] = config_dict["gl_version"]
        settings.WINDOW['size'] = config_dict["window_size"]# Adjust this to your desired window size
        settings.WINDOW['aspect_ratio'] = config_dict["window_size"][0] / config_dict["window_size"][1]
        settings.WINDOW["fullscreen"] = config_dict["fullscreen"]
        settings.WINDOW["samples"] = 0

        self.aspect_ratio = config_dict["window_size"][0] / config_dict["window_size"][1]

        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (config_dict["y_shift"], config_dict["x_shift"])
        self.window.init_mgl_context()
        self.stop = False
        self.window.set_default_viewport()

    def run(self):
        while not self.window.is_closing:
            self.window.use()
            self.window.ctx.clear(0.5, 0.5, 0.5, 1.0)
            self.window.swap_buffers()
            self.communicate()
            time.sleep(0.001)
        self.window.close()

    def communicate(self):
        if not self.queue.empty():
            command = self.queue.get()
            if Path(command).suffix == ".h5":
                self.play_noise(command)
            elif command == "stop":
                self.stop = False
                self.run()
    def play_noise(self, file):
        all_patterns_3d, width, height, frames, desired_fps  = load_3d_patterns(file)

        patterns = [
            self.window.ctx.texture((width, height), 1, all_patterns_3d[i].tobytes())
            for i in range(frames)
        ]
        program = self.window.ctx.program(vertex_shader=open('vertex_shader.glsl').read(),
                                          fragment_shader=open("fragment_shader.glsl").read())


        #TODO Set aspect ratio correctly
        window_width, window_height = self.window.size
        window_aspect = window_width / window_height
        texture_aspect = width / height

        if texture_aspect > window_aspect:
            texture_scale_x = 1.0
            texture_scale_y = (height / width) * window_aspect
        else:
            texture_scale_x = (width / height) / window_aspect
            texture_scale_y = 1.0
        program['aspect_adjustment'].value = window_aspect
        program['scale'].value = (texture_scale_x, texture_scale_y)

        quad = np.array([
            -1.0, 1.0,  # top left
            -1.0, -1.0,  # bottom left
            1.0, 1.0,  # top right
            1.0, 1.0,  # top right
            -1.0, -1.0,  # bottom left
            1.0, -1.0  # bottom right
        ], dtype=np.float32)

        vbo = self.window.ctx.buffer(quad.tobytes())
        vao = self.window.ctx.simple_vertex_array(program, vbo, 'in_pos')


        time_per_frame = 1.0 / desired_fps
        last_update = time.time()
        current_pattern_index = 0
        while not self.window.is_closing:

            self.communicate()

            self.window.use()  # Ensure the correct context is being used
            start_time = time.time()  # Start time for this frame
            elapsed_time = start_time - last_update

            if elapsed_time >= time_per_frame:
                # Clear the window
                self.window.ctx.clear(0.5, 0.5, 0.5)

                # Bind the texture to texture unit 0
                patterns[current_pattern_index].use(location=0)

                # Set the shader uniform to the index of the texture unit

                program['pattern'].value = 0

                # Render the full screen quad
                vao.render(moderngl.TRIANGLES)

                # Swap buffers after rendering
                self.window.swap_buffers()

                # Measure frame duration
                frame_duration = time.time() - start_time
                if frame_duration > time_per_frame:
                    print(
                        f"WARNING: Frame duration of {frame_duration * 1000:.2f} ms exceeds the desired {time_per_frame * 1000:.2f} ms!")

                # Update the last update time
                last_update = start_time
                current_pattern_index += 1
                if current_pattern_index == len(patterns) - 1:
                    self.run()
            else:
                # Sleep for a short duration to avoid busy waiting
                time.sleep(0.001)  # Sleep for 1 millisecond



def load_3d_patterns(file):
    with h5py.File(f"stimuli/{file}", 'r') as f:
        noise = f['Noise'][:]
        frame_rate = f['Frame_Rate'][()]
    size = noise.shape
    width = size[1]
    height = size[2]
    frames = size[0]
    return noise, width, height, frames, frame_rate




def pyglet_app(config, queue):
    Noise = Presenter(config, queue)
    Noise.run()


if __name__ == '__main__':
    pyglet_app(Queue())









