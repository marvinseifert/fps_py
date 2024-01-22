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


class Presenter:
    """
    This class is responsible for presenting the stimuli. It is a wrapper around the pyglet window class and the
    moderngl_window BaseWindow class. It is responsible for loading the noise stimuli and presenting them. It is also
    responsible for communicating with the main process (gui) via a queue.

    """

    delay = 10

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
        mode,
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
        self.nr_followers = len(config_dict["windows"].keys()) - 1
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

    def __del__(self):
        with self.ard_lock:
            self.ard_queue.put("destroy")

    def run_empty(self):
        """
        Empty loop. Establishes a window filled with a grey background. Waits for commands from the main process (gui).


        """
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
            if type(command) == dict:
                self.play_noise(command)
            elif command == "white_screen":
                self.play_white()
            elif command == "stop":  # If the command is "stop", stop the presentation
                self.stop = False  # Trigger the stop flag for next time
                self.send_colour("O")
                self.run_empty()  # Run the empty loop
            elif command == "destroy":
                self.window.close()  # Close the window

    def send_array(self, array):
        """Send array string of shared memory to other processes"""
        for _ in range(self.nr_followers):
            self.sync_queue.put(array)

    def receive_array(self):
        """Receive array string of shared memory from lead process"""
        return self.sync_queue.get()

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        with self.ard_lock:
            self.ard_queue.put("T")

    def send_colour(self, colour):
        """Send a colour signal to the Arduino."""
        with self.ard_lock:
            self.ard_queue.put(colour)

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

    def process_colours(self, colours, change_logic, frames):
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
        all_patterns_3d, width, height, frames, desired_fps = load_3d_patterns(
            file
        )  # Load the noise data

        # Establish the texture for each noise frame
        patterns = [
            self.window.ctx.texture(
                (width, height),
                1,
                all_patterns_3d[i, :, :].tobytes(),
                samples=0,
                alignment=1,
            )
            for i in range(frames)
        ]

        return all_patterns_3d, width, height, frames, desired_fps, patterns

    def setup_shader_program(self):
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
        with open("fragment_shader.glsl", "r") as fragment_file:
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
        self, pattern_indices, s_frames, colours, change_logic, patterns, program, vao
    ):
        """
        Main loop for presenting the noise.

        Parameters
        ----------
        pattern_indices : list
            List of indices indicating the order in which to present the patterns.
        s_frames : list
            List of timestamps for when each frame should start.
        colours : list
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
        last_update = time.perf_counter()
        time_per_frame = 1 / len(
            pattern_indices
        )  # Assuming frames are equally distributed

        for idx, current_pattern_index in enumerate(pattern_indices):
            self.communicate()  # Custom function for communication, can be modified as needed

            # Sync frame presentation to the scheduled time
            while time.perf_counter() < s_frames[idx]:
                pass  # Busy-wait until the scheduled frame time
            start_time = time.perf_counter()

            self.window.use()  # Ensure the correct context is being used

            # Handle colour change logic
            if current_pattern_index % change_logic == 0:
                c = colours[current_pattern_index // change_logic % len(colours)]
                self.send_colour(c)  # Custom function to send colour to Arduino

            # Clear the window and render the noise
            self.window.ctx.clear(0, 0, 0)
            patterns[current_pattern_index].use(location=0)
            program[
                "pattern"
            ].value = 0  # Assuming 'pattern' is the uniform name in shader
            vao.render(moderngl.TRIANGLES)

            # Swap buffers and send trigger signal
            self.window.swap_buffers()
            self.send_trigger()  # Custom function to send a trigger signal to Arduino

            # Monitor and log frame duration, if necessary
            frame_duration = time.perf_counter() - start_time
            last_update = time.perf_counter()
            if frame_duration > (time_per_frame + self.frame_duration):
                print(
                    f"WARNING: Frame duration exceeded. Duration: {frame_duration:.2f}s"
                )

            # Break the loop if the last frame was presented
            if idx >= len(pattern_indices) - 1:
                break

    def cleanup_and_finalize(self, patterns, vbo, vao, noise_dict):
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
        """
        # Send final colour signal or perform any final communication
        self.send_colour("O")  # Assuming 'O' is the signal for completion

        # Release all pattern textures
        for pattern in patterns:
            pattern.release()

        # Release the buffer and vertex array object
        vbo.release()
        vao.release()

        # Write log with the noise_dict or any other relevant information
        write_log(noise_dict)  # Assuming 'write_log' is a function for logging

        # Run any additional emptying or resetting procedures
        self.run_empty()  # Assuming 'run_empty' is a method for final procedures

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
        file, loops, colours, change_logic, s_frames = self.load_and_initialize_data(
            noise_dict
        )

        colours = self.process_colours(colours, change_logic, len(s_frames))

        # Load the noise data
        (
            all_patterns_3d,
            width,
            height,
            frames,
            desired_fps,
            patterns,
        ) = self.load_noise_data(file)

        # Establish the texture for each noise frame
        patterns = [
            self.window.ctx.texture(
                (width, height),
                1,
                all_patterns_3d[i, :, :].tobytes(),
                samples=0,
                alignment=1,
            )
            for i in range(frames)
        ]
        # Establish the shader program for presenting the noise
        program = self.setup_shader_program()

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
        s_frames = [s + self.delay for s in s_frames]

        print(f"stimulus will start in {s_frames[0] - time.perf_counter()} seconds")

        # Start the presentation loop
        self.presentation_loop(
            pattern_indices, s_frames, colours, change_logic, patterns, program, vao
        )

        # Clean up and finalize the presentation
        self.cleanup_and_finalize(patterns, vbo, vao, noise_dict)

    def play_white(self):
        """
        This function presents a white screen until a stop command is received.
        """

        while not self.window.is_closing:
            self.window.use()
            self.window.ctx.clear(1, 1, 1)
            self.window.swap_buffers()
            self.communicate()
            time.sleep(0.001)
        self.window.close()


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
    filename_format = (
        f"logs/{file}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')}"
    )

    with open(filename_format, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_file", "loops", "colours", "change_logic", "time"])
        writer.writerow([file, loops, colours, change_logic, time.strftime("%H:%M:%S")])


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
    with h5py.File(f"stimuli/{file}", "r") as f:
        noise = np.asarray(f["Noise"][:], dtype=np.uint8)
        frame_rate = f["Frame_Rate"][()]

    size = noise.shape
    print(noise.dtype)
    width = size[2]
    height = size[1]
    frames = size[0]
    # noise = np.asfortranarray(noise)

    return noise, width, height, frames, frame_rate


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
    mode="lead",
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
        mode,
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
    )
    Noise.run_empty()  # Establish the empty loop


# Can run the pyglet app from here for testing purposes if needed
# if __name__ == '__main__':
#     pyglet_app(Queue())
