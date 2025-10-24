import moderngl_window
import moderngl
from moderngl_window.conf import settings
import time
import h5py
import numpy as np
from pathlib import Path
import csv
import datetime
from multiprocessing import RawArray
from multiprocessing import sharedctypes
import pyglet
from arduino import Arduino
import threading
import pydevd_pycharm

class Presenter:
    """
    This class is responsible for presenting the stimuli. It is a wrapper around the pyglet window class and the
    moderngl_window BaseWindow class. It is responsible for loading the noise stimuli and presenting them. It is also
    responsible for communicating with the main process (gui) via a queue.

    """

    def __init__(
        self,
        process_idx,
        config_dict,
        queue,
        sync_queue,
        sync_lock,
        lock,
        ard_queue,
        ard_lock,
        status_queue,
        status_lock,
        mode,
        delay=10,
    ):
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
        self.process_idx = process_idx
        self.queue = queue
        self.sync_queue = sync_queue
        self.sync_lock = sync_lock
        self.lock = lock
        self.mode = mode
        self.ard_queue = ard_queue
        self.ard_lock = ard_lock
        self.status_queue = status_queue
        self.status_lock = status_lock
        self.nr_followers = len(config_dict["windows"].keys()) - 1
        self.c_channels = config_dict["windows"][str(self.process_idx)]["channels"]
        self.delay = delay
        self.arduino_running = False
        settings.WINDOW[
            "class"
        ] = "moderngl_window.context.pyglet.Window"  # using a pyglet window
        settings.WINDOW["gl_version"] = config_dict["gl_version"]
        settings.WINDOW["size"] = config_dict["windows"][str(self.process_idx)][
            "window_size"
        ]
        settings.WINDOW[
            "aspect_ratio"
        ] = None  # Sets the aspect ratio to the window's aspect ratio
        settings.WINDOW["fullscreen"] = config_dict["windows"][str(self.process_idx)][
            "fullscreen"
        ]
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True
        settings.WINDOW["resizable"] = False
        settings.WINDOW["title"] = "Noise Presentation"
        settings.WINDOW["style"] = config_dict["windows"][str(self.process_idx)][
            "style"
        ]

        self.frame_duration = 1 / config_dict["fps"]  # Calculate the frame duration

        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (
            config_dict["windows"][str(self.process_idx)]["x_shift"],
            config_dict["windows"][str(self.process_idx)]["y_shift"],
        )  # Shift the window
        self.window.init_mgl_context()  # Initialize the moderngl context
        self.stop = False  # Flag for stopping the presentation
        self.window.set_default_viewport()  # Set the viewport to the window size

        if self.mode == "lead":
            self.arduino = Arduino(
                port=config_dict["windows"][str(self.process_idx)]["arduino_port"],
                baud_rate=config_dict["windows"][str(self.process_idx)][
                    "arduino_baud_rate"
                ],
                queue=ard_queue,
                queue_lock=ard_lock,
            )

    def __del__(self):
        with self.ard_lock:
            if self.mode == "lead":
                self.arduino.disconnect()

    def run_empty(self):
        """
        Empty loop. Establishes a window filled with a grey background. Waits for commands from the main process (gui).


        """

        # if self.mode == "lead":
        #     self.arduino.send("W")
        while not self.window.is_closing:
            self.window.use()
            # self.window.ctx.clear(0.5, 0.5, 0.5, 1.0)  # Clear the window with a grey background
            self.window.ctx.clear(0, 0, 0, 1.0)

            self.window.swap_buffers()  # Swap the buffers (update the window content)
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.window.close()  # Close the window in case it is closed by the user

    def communicate(self):
        """
        Check for commands from the main process (gui). If a command is found, execute it.
        """
        command = None
        with self.lock:
            if not self.queue.empty():
                command = self.queue.get()

        if command:
            if type(command) == dict:  # This would be an array to play.
                self.stop = False
                self.play_noise(command)
            elif command == "white_screen":
                self.stop = False
                if self.mode == "lead":
                    with self.ard_lock:
                        ard_command = self.ard_queue.get()
                        self.send_colour(ard_command)
                        self.arduino_running = True
                        arduino_thread = threading.Thread(
                            target=self.receive_arduino_status
                        )
                        arduino_thread.start()

            elif command == "stop":  # If the command is "stop", stop the presentation
                self.arduino_running = False  # Trigger the stop flag for next time
                self.status_queue.put("done")
                if self.mode == "lead":
                    self.send_colour("b")
                    self.send_colour("b")
                    self.send_colour("b")
                    self.send_colour("O")
                self.stop = True
                current_time = time.perf_counter()
                while time.perf_counter() - current_time < 1:
                    pass
            elif command == "destroy":
                self.window.close()  # Close the window

    def receive_arduino_status(self):
        buffer = True
        self.arduino.arduino.reset_input_buffer()
        while not self.stop:
            status = self.arduino.read()
            if status == "Trigger":
                buffer = False
            if status == "finished" and not buffer:
                self.arduino_running = False
                self.status_queue.put("done")
                break

    def send_array(self, array):
        """Send array string of shared memory to other processes"""
        for _ in range(self.nr_followers):
            self.sync_queue.put(array)

    def receive_array(self):
        """Receive array string of shared memory from lead process"""
        return self.sync_queue.get()

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        if self.mode == "lead":
            self.arduino.send("T")

    def switch_trigger_modes(self, mode="t_s_off"):
        """Switch the trigger mode of the Arduino."""
        if self.mode == "lead":
            self.arduino.send(mode)

    def send_colour(self, colour):
        """Send a colour signal to the Arduino."""
        if self.mode == "lead":
            self.arduino.send(colour)

    def load_and_initialize_data(self, noise_dict):
        """
        Load and initialize data from the provided dictionary.

        Parameters
        ----------
        noise_dict : dict
            A dictionary containing various settings for noise playback.

        Returns
        -------
        tuple
            A tuple containing initialized values: file, loops, colours,
            change_logic, s_frames.
        """
        # Extract parameters from the noise_dict
        file = noise_dict["file"]
        loops = noise_dict["loops"]
        colours = noise_dict["colours"]
        change_logic = noise_dict["change_logic"]
        s_frames_temp = noise_dict["s_frames"]

        # Copy and modify s_frames based on loops
        s_frames = s_frames_temp.copy()
        first_frame_dur = np.diff(s_frames[0:2])
        for loop in range(1, loops):
            s_frames = np.concatenate(
                (
                    s_frames,
                    s_frames_temp
                    + loop * (s_frames_temp[-1] - s_frames_temp[0] + first_frame_dur),
                )
            )

        return file, loops, colours, change_logic, s_frames

    def process_arduino_colours(self, colours, change_logic, frames):
        """
        Processes the colours and calculates the necessary repeats.

        Parameters
        ----------
        colours : str
            A comma-separated string of colour values.
        change_logic : int
            The logic determining how often the colour changes.
        frames : int
            The total number of frames for the noise.

        Returns
        -------
        list
            A list of colours repeated and arranged as per the specified logic.
        """
        colours = colours.split(",")
        colour_repeats = np.ceil(frames / (len(colours) * change_logic))
        colours = np.repeat(np.asarray(colours), change_logic).tolist()
        colours = colours * int(colour_repeats)

        return colours

    def load_noise_data(self, file):
        """
        Loads the noise data from a file and establishes textures for each noise frame.

        Parameters
        ----------
        file : str
            The path to the noise file.

        Returns
        -------
        tuple
            A tuple containing the loaded patterns as a 3D array, the width and height of each pattern,
            the number of frames, and the desired frames per second (fps).
        """
        (
            all_patterns_3d,
            width,
            height,
            frames,
            desired_fps,
            nr_colours,
        ) = load_3d_patterns(
            file, channels=self.c_channels
        )  # Load the noise data

        # Establish the texture for each noise frame
        patterns = [
            self.window.ctx.texture(
                (width, height),
                nr_colours,
                all_patterns_3d[i, :].tobytes(),
                samples=0,
                alignment=1,
            )
            for i in range(frames)
        ]

        return all_patterns_3d, width, height, frames, desired_fps, patterns, nr_colours

    def setup_shader_program(self, nr_colours=1):
        """
        Initializes the shader program using vertex and fragment shaders.

        Returns
        -------
        moderngl.Program
            The compiled and linked shader program.
        """
        # Load and compile vertex and fragment shaders
        with open("vertex_shader.glsl", "r") as vertex_file:
            vertex_shader_source = vertex_file.read()
        if nr_colours == 1:
            with open("fragment_shader.glsl", "r") as fragment_file:
                fragment_shader_source = fragment_file.read()
        else:
            with open("fragment_shader_colour.glsl", "r") as fragment_file:
                fragment_shader_source = fragment_file.read()

        # Create and return the shader program
        program = self.window.ctx.program(
            vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
        )

        return program

    def calculate_scaling(self, width, height):
        """
        Calculates the scaling factors based on the aspect ratio of the texture and the window.

        Parameters
        ----------
        width : int
            The width of the texture.
        height : int
            The height of the texture.

        Returns
        -------
        tuple
            A tuple containing the scaling factors (scale_x, scale_y) and the quad vertices array.
        """
        # Calculate the aspect ratio of the window and the texture
        window_width, window_height = self.window.size
        window_aspect = window_width / window_height
        texture_aspect = width / height

        # Determine scaling factors based on aspect ratios

        scale_x = width / window_width
        scale_y = height / window_height
        if (window_aspect == texture_aspect) & (window_aspect > 1):
            scale_x = scale_y = 1

        # Establish the vertices for the texture in the shader program
        quad = np.array(
            [
                -scale_x,
                scale_y,  # top left
                -scale_x,
                -scale_y,  # bottom left
                scale_x,
                scale_y,  # top right
                scale_x,
                scale_y,  # top right
                -scale_x,
                -scale_y,  # bottom left
                scale_x,
                -scale_y,  # bottom right
            ],
            dtype=np.float32,
        )

        return scale_x, scale_y, quad

    def create_buffer_and_vao(self, quad, program):
        """
        Creates a buffer and vertex array object (VAO) for rendering.

        Parameters
        ----------
        quad : np.array
            Array of vertices for the quad.
        program : moderngl.Program
            The shader program used for rendering.

        Returns
        -------
        moderngl.Buffer
            The created vertex buffer object (VBO).
        moderngl.VertexArray
            The created vertex array object (VAO).
        """
        # Create a buffer from the quad vertices
        vbo = self.window.ctx.buffer(quad.tobytes())

        # Create a vertex array object
        vao = self.window.ctx.simple_vertex_array(program, vbo, "in_pos")

        return vbo, vao

    def setup_presentation(self, frames, loops, desired_fps):
        """
        Sets up the presentation parameters including time per frame and pattern indices.

        Parameters
        ----------
        frames : int
            The number of frames in the noise pattern.
        loops : int
            The number of times the noise pattern should loop.
        desired_fps : float
            The desired frames per second for the presentation.

        Returns
        -------
        float
            The time allocated per frame.
        list
            The list of pattern indices for the presentation loop.
        """
        # Calculate the time per frame for the desired FPS
        time_per_frame = 1 / desired_fps

        # Calculate the pattern indices for each frame in the loop
        pattern_indices = np.arange(frames)  # Generate indices for each frame
        pattern_indices = np.tile(
            pattern_indices, loops
        )  # Repeat indices for each loop

        return time_per_frame, pattern_indices.tolist()

    import time

    def presentation_loop(
        self,
        pattern_indices,
        s_frames,
        end_times,
        nr_colours,
        arduino_colours,
        change_logic,
        patterns,
        program,
        vao,
    ):
        """
        Main loop for presenting the noise.

        Parameters
        ----------
        pattern_indices : list
            List of indices indicating the order in which to present the patterns.
        s_frames : list
            List of timestamps for when each frame should start.
        arduino_colours : list
            List of colours to be used for each frame.
        change_logic : int
            Logic to determine when to change the colour.
        patterns : list
            List of textures for each noise frame.
        program : moderngl.Program
            The shader program used for rendering.
        vao : moderngl.VertexArray
            The vertex array object for rendering.
        """

        for idx, current_pattern_index in enumerate(pattern_indices):
            self.communicate()  # Custom function for communication, can be modified as needed
            if self.stop:
                del patterns
                del program
                del vao
                return end_times
            # Sync frame presentation to the scheduled time
            while time.perf_counter() < s_frames[idx]:
                pass  # Busy-wait until the scheduled frame time

            self.window.use()  # Ensure the correct context is being used

            # Handle colour change logic
            if current_pattern_index % change_logic == 0:
                c = arduino_colours[current_pattern_index]

                self.send_colour(c)  # Custom function to send colour to Arduino

            # Clear the window and render the noise
            self.window.ctx.clear(0, 0, 0)
            patterns[current_pattern_index].use(location=0)
            if nr_colours > 1:
                program[
                    "pattern"
                ].red = 0  # Assuming 'pattern' is the uniform name in shader
                program["pattern"].green = 0
                program["pattern"].blue = 0
            else:
                program["pattern"].value = 0
            vao.render(moderngl.TRIANGLES)

            # Swap buffers and send trigger signal
            start_time = time.perf_counter()
            self.window.swap_buffers()
            self.send_trigger()  # Custom function to send a trigger signal to Arduino

            # Monitor and log frame duration, if necessary
            end_times[idx] = time.perf_counter() - start_time

            # Break the loop if the last frame was presented
            if idx >= len(pattern_indices) - 1:
                return end_times

    def cleanup_and_finalize(
        self, patterns, vbo, vao, noise_dict, end_times, desired_fps
    ):
        """
        Cleans up resources, writes logs, and runs final procedures after the presentation.

        Parameters
        ----------
        patterns : list
            List of texture objects to be released.
        vbo : moderngl.Buffer
            The vertex buffer object to be released.
        vao : moderngl.VertexArray
            The vertex array object to be released.
        noise_dict : dict
            The dictionary containing noise settings, used for logging purposes.
        end_times : list
            List of frame durations.
        desired_fps : float
            The desired frames per second for the presentation.
        """
        # Send final colour signal or perform any final communication
        self.send_colour("O")  # Assuming 'O' is the signal for completion

        # Release all pattern textures
        for pattern in patterns:
            pattern.release()
        del patterns
        # Release the buffer and vertex array object
        vbo.release()
        vao.release()
        del vbo
        del vao

        # Check which frames were dropped
        dropped_frames = np.where(end_times - (1 / desired_fps) > 0)
        wrong_frame_times = end_times[dropped_frames[0]]

        if len(dropped_frames[0]) == 0:
            dropped_frames = None
            wrong_frame_times = None
        else:
            # Print the dropped frames
            print(f"dropped frames (idx): {dropped_frames[0]}")
            print(f"wrong frame times: {wrong_frame_times}")

            # Write log with the noise_dict or any other relevant information
        write_log(
            noise_dict, dropped_frames, wrong_frame_times
        )  # Assuming 'write_log' is a function for logging

        # Run any additional emptying or resetting procedures
        self.stop = False
        return

    def play_noise(self, noise_dict):
        """
        Play the noise file. This function loads the noise file, creates a texture from it and presents it.
        Parameters
        ----------
        file : str

            Path to the noise file.

        """
        # p
        # Get the data according to the noise_dict
        (
            file,
            loops,
            arduino_colours,
            change_logic,
            s_frames,
        ) = self.load_and_initialize_data(noise_dict)

        arduino_colours = self.process_arduino_colours(
            arduino_colours, change_logic, len(s_frames) - 1
        )

        # Load the noise data
        (
            all_patterns_3d,
            width,
            height,
            frames,
            desired_fps,
            patterns,
            nr_colours,
        ) = self.load_noise_data(file)

        # Establish the shader program for presenting the noise
        program = self.setup_shader_program(nr_colours)

        # Calculate the aspect ratio of the window and the noise to adjust the noise size
        scale_x, scale_y, quad = self.calculate_scaling(width, height)

        # Create the buffer and vertex array object for the noise
        vbo, vao = self.create_buffer_and_vao(quad, program)

        # Establish the time per frame for the desired fps

        time_per_frame, pattern_indices = self.setup_presentation(
            frames, loops, desired_fps
        )

        # Synchronize the presentation

        # Add buffer delay to frames:
        delay_needed = s_frames[0] - time.perf_counter()
        if delay_needed > 0:
            delay = 10
        else:
            delay = np.abs(delay_needed) + 10
        s_frames = s_frames + delay

        print(
            f"stimulus will start in {s_frames[0] - time.perf_counter()} seconds, window_idx: {self.process_idx}"
        )
        print(f"Current time is {datetime.datetime.now()}")
        end_times = np.zeros(len(s_frames))
        # Start the presentation loop
        self.switch_trigger_modes("t_s_on")
        end_times = self.presentation_loop(
            pattern_indices,
            s_frames,
            end_times,
            nr_colours,
            arduino_colours,
            change_logic,
            patterns,
            program,
            vao,
        )
        self.switch_trigger_modes("t_s_off")

        # Clean up and finalize the presentation
        self.cleanup_and_finalize(
            patterns, vbo, vao, noise_dict, end_times, desired_fps
        )


def write_log(noise_dict, dropped_frames=None, wrong_frame_times=None):
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
    filename_format = (
        f"logs/{file}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')}"
    )

    if dropped_frames is None:
        dropped_frames = []
        wrong_frame_times = []

    with open(filename_format, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "noise_file",
                "loops",
                "colours",
                "change_logic",
                "time",
                "dropped_frames",
                "wrong_frame_times",
            ]
        )
        writer.writerow(
            [
                file,
                loops,
                colours,
                change_logic,
                time.strftime("%H:%M:%S"),
                dropped_frames,
                wrong_frame_times,
            ]
        )


def load_3d_patterns(file, channels=None):
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
    with h5py.File(f"stimuli/{file}", "r") as f:
        noise = np.asarray(f["Noise"][:], dtype=np.uint8)
        frame_rate = f["Frame_Rate"][()]

    size = noise.shape
    width = size[2]
    height = size[1]
    frames = size[0]
    try:
        colours = size[3]
    except IndexError:
        colours = 1

    if (colours > 1) & (channels is not None):
        try:
            noise = noise[:, :, :, channels]
            colours = len(channels)
        except IndexError:
            print("more channels requested than available in the noise file")
            raise
    # noise = np.asfortranarray(noise)

    return noise, width, height, frames, frame_rate, colours


def get_noise_info(file):
    with h5py.File(f"stimuli/{file}", "r") as f:
        noise = f["Noise"][:]
        frame_rate = f["Frame_Rate"][()]
    size = noise.shape
    width = size[2]
    height = size[1]
    frames = size[0]
    return width, height, frames, frame_rate


def pyglet_app_lead(
    process_idx,
    config,
    queue,
    sync_queue,
    sync_lock,
    lock,
    ard_queue,
    ard_lock,
    status_queue,
    status_lock,
    delay=10,
):
    """
    Start the pyglet app. This function is used to spawn the pyglet app in a separate process.
    Parameters
    ----------
    process_idx : int
        Index of the process. Used to determine the window position.
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
    Noise = Presenter(
        process_idx,
        config,
        queue,
        sync_queue,
        sync_lock,
        lock,
        ard_queue,
        ard_lock,
        status_queue,
        status_lock,
        mode="lead",
        delay=delay,
    )
    Noise.run_empty()  # Establish the empty loop


def pyglet_app_follow(
    process_idx,
    config,
    queue,
    sync_queue,
    sync_lock,
    lock,
    ard_queue,
    ard_lock,
    delay=10,
):
    """
    Start the pyglet app. This function is used to spawn the pyglet app in a separate process.
    Parameters
    ----------
        process_idx : int
        Index of the process. Used to determine the window position.
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
    Noise = Presenter(
        process_idx,
        config,
        queue,
        sync_queue,
        sync_lock,
        lock,
        ard_queue,
        ard_lock,
        mode="follow",
        delay=delay,
    )
    Noise.run_empty()  # Establish the empty loop


# Can run the pyglet app from here for testing purposes if needed
# if __name__ == '__main__':
#     pyglet_app(Queue())
