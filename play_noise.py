import moderngl_window
import moderngl
from moderngl_window.conf import settings
import time
import h5py
import numpy as np
from pathlib import Path
import serial
import csv
import datetime


def connect_to_arduino(port='COM3', baud_rate=9600):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate)
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None


class Presenter:
    """
    This class is responsible for presenting the stimuli. It is a wrapper around the pyglet window class and the
    moderngl_window BaseWindow class. It is responsible for loading the noise stimuli and presenting them. It is also
    responsible for communicating with the main process (gui) via a queue.

    """

    def __init__(self, config_dict, queue):
        """
        Parameters
        ----------
        config_dict : dict
            Dictionary containing the configuration parameters for the window.
            Keys:
                "y_shift" : int
                    Shift of the window in y direction.
                "x_shift" : int
                    Shift of the window in x direction.
                "gl_version" : tuple
                    Version of OpenGL to use.
                "window_size" : tuple
                    Size of the window.
                "fullscreen" : bool
                    Whether to use fullscreen mode or not. Fullscreen is currently only working on the
                    main monitor.

        queue : multiprocessing.Queue
            Queue for communication with the main process (gui).

        """

        self.queue = queue
        settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'  # using a pyglet window
        settings.WINDOW['gl_version'] = config_dict["gl_version"]
        settings.WINDOW['size'] = config_dict["window_size"]
        settings.WINDOW['aspect_ratio'] = None  # Sets the aspect ratio to the window's aspect ratio
        settings.WINDOW["fullscreen"] = config_dict["fullscreen"]
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True

        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (config_dict["x_shift"], config_dict["y_shift"])  # Shift the window
        self.window.init_mgl_context()  # Initialize the moderngl context
        self.stop = False  # Flag for stopping the presentation
        self.window.set_default_viewport()  # Set the viewport to the window size
        self.arduino = connect_to_arduino()  # Establish a connection to the Arduino

    def __del__(self):
        if self.arduino:
            self.send_colour("O")
            self.arduino.close()

    def run_empty(self):
        """
        Empty loop. Establishes a window filled with a grey background. Waits for commands from the main process (gui).


        """
        while not self.window.is_closing:
            self.window.use()
            self.window.ctx.clear(0.5, 0.5, 0.5, 1.0)  # Clear the window with a grey background
            self.window.swap_buffers()  # Swap the buffers (update the window content)
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.window.close()  # Close the window in case it is closed by the user

    def communicate(self):
        """
        Check for commands from the main process (gui). If a command is found, execute it.
        """
        if not self.queue.empty():
            command = self.queue.get()
            if type(command) == dict:
                self.play_noise(command)
            elif command == "stop":  # If the command is "stop", stop the presentation
                self.stop = False  # Trigger the stop flag for next time
                self.send_colour("O")
                self.run_empty()  # Run the empty loop
            elif command == "destroy":
                self.window.close()  # Close the window

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        if self.arduino:
            try:
                self.arduino.write(b'T')  # Sending a 'T' as the trigger. Modify as needed.
            except Exception as e:
                print(f"Error sending trigger to Arduino: {e}")

    def send_colour(self, colour):
        if self.arduino:
            try:
                txt = f"{colour}".encode()  # Convert the colour string to bytes
                self.arduino.write(txt)
            except Exception as e:
                print(f"Error sending trigger to Arduino: {e}")

    def play_noise(self, noise_dict):
        """
        Play the noise file. This function loads the noise file, creates a texture from it and presents it.
        Parameters
        ----------
        file : str

            Path to the noise file.

        """
        file = noise_dict["file"]
        loops = noise_dict["loops"]
        colours = noise_dict["colours"]
        change_logic = noise_dict["change_logic"]

        colours = colours.split(",")
        # colours = [x for x in colours for _ in range(change_logic)]

        all_patterns_3d, width, height, frames, desired_fps = load_3d_patterns(file)  # Load the noise data

        colour_repeats = int(np.ceil(frames / change_logic / len(colours)))
        colours = colours * colour_repeats

        # Establish the texture for each noise frame
        patterns = [
            self.window.ctx.texture((width, height), 1, all_patterns_3d[i, :, :].tobytes(),
                                    samples=0, alignment=1)
            for i in range(frames)
        ]
        # Establish the shader program for presenting the noise
        program = self.window.ctx.program(vertex_shader=open('vertex_shader.glsl').read(),
                                          fragment_shader=open("fragment_shader.glsl").read())

        # Calculate the aspect ratio of the window and the noise to adjust the noise size
        window_width, window_height = self.window.size
        window_aspect = window_width / window_height
        texture_aspect = width / height

        if window_aspect != texture_aspect:
            scale_x = width / window_width
            scale_y = height / window_height
        else:
            scale_x = 1
            scale_y = 1

        # Establish the vertices for the noise used in the shader program

        quad = np.array([
            -scale_x, scale_y,  # top left
            -scale_x, -scale_y,  # bottom left
            scale_x, scale_y,  # top right
            scale_x, scale_y,  # top right
            -scale_x, -scale_y,  # bottom left
            scale_x, -scale_y  # bottom right
        ], dtype=np.float32)

        # Create the buffer and vertex array object for the noise
        vbo = self.window.ctx.buffer(quad.tobytes())
        vao = self.window.ctx.simple_vertex_array(program, vbo, 'in_pos')

        # Establish the time per frame for the desired fps
        time_per_frame = 1.0 / desired_fps
        last_update = time.time()

        # Main loop for presenting the noise
        while not self.window.is_closing:

            for loop in range(loops):

                # Check for commands from the main process (gui) (for example stop commands)
                current_pattern_index = 0
                c_index = 0
                c = colours[c_index]

                while current_pattern_index <= frames:

                    self.communicate()

                    self.window.use()  # Ensure the correct context is being used
                    start_time = time.time()  # Start time for this frame
                    elapsed_time = start_time - last_update

                    if elapsed_time >= time_per_frame:  # Wait for correct time to present the next frame

                        if current_pattern_index % change_logic == 0:
                            if current_pattern_index != 0:
                                c_index += 1
                                c = colours[c_index]

                        self.send_colour(c)  # Sends colour to Arduino

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
                        self.send_trigger()  # Send a trigger signal to the Arduino

                        # Measure frame duration
                        frame_duration = time.time() - start_time
                        if frame_duration > time_per_frame:
                            # If the frame duration exceeds the desired frame duration, print a warning
                            print(
                                f"WARNING: Frame duration of {frame_duration * 1000:.2f} ms exceeds the desired {time_per_frame * 1000:.2f} ms!")

                        # Update the last update time
                        last_update = start_time
                        current_pattern_index += 1
                        # If the last frame was presented, stop the presentation
                        if current_pattern_index > len(patterns) - 1:
                            # self.run_empty()
                            break
                    else:
                        # Sleep for a short duration to avoid busy waiting
                        time.sleep(0.001)  # Sleep for 1 millisecond
            self.send_colour("O")
            for pattern in patterns:
                pattern.release()
            vbo.release()
            vao.release()
            write_log(noise_dict)
            self.run_empty()


def write_log(noise_dict):
    """
    Write the log file for the noise presentation.
    Parameters
    ----------
    noise_dict : str
        Path to the noise file.
    """
    file = noise_dict["file"]
    loops = noise_dict["loops"]
    colours = noise_dict["colours"]
    change_logic = noise_dict["change_logic"]
    filename_format = f"logs/{file}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')}"

    with open(filename_format, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["noise_file", "loops", "colours", "change_logic", "time"])
        writer.writerow([file, loops, colours, change_logic, time.strftime('%H:%M:%S')])


def load_3d_patterns(file):
    """
    Load the noise .h5 file and return the noise data, width, height, frames and frame rate.
    Parameters
    ----------
    file : str
        Path to the noise file.
    Returns
    -------
    noise : np.ndarray
        Noise data.
    width : int
        Width of the noise.
    height : int
        Height of the noise.
    frames : int
        Number of frames in the noise.
    frame_rate : int
        Frame rate of the noise.

    """
    with h5py.File(f"stimuli/{file}", 'r') as f:
        noise = f['Noise'][:]
        frame_rate = f['Frame_Rate'][()]
    size = noise.shape
    width = size[2]
    height = size[1]
    frames = size[0]
    # noise = np.asfortranarray(noise)

    return noise, width, height, frames, frame_rate


def pyglet_app(config, queue):
    """
    Start the pyglet app. This function is used to spawn the pyglet app in a separate process.
    Parameters
    ----------
    config : dict
        Configuration dictionary.
        Keys:
            width : int
                Width of the window.
            height : int
                Height of the window.
            fullscreen : bool
                Fullscreen mode.
            screen : int
                Screen number.
    queue : multiprocessing.Queue
        Queue for communication with the main process (gui).
    """
    Noise = Presenter(config, queue)
    Noise.run_empty()  # Establish the empty loop

# Can run the pyglet app from here for testing purposes if needed
# if __name__ == '__main__':
#     pyglet_app(Queue())
