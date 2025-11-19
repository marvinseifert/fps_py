import csv
import datetime
from pathlib import Path
import threading
import time
import h5py
import numpy as np
import moderngl
import moderngl_window
from moderngl_window.conf import settings
import pyglet
from arduino import Arduino, DummyArduino
import threading
import importlib.resources
import fpspy.arduino
import fpspy.config
import fpspy.queue

# import pydevd_pycharm


def _loop(frames, loops):
    """Loop the frames array a given number of times."""
    frames_cpy = frames.copy()
    first_frame_dur = np.diff(frames[0:2])
    for l in range(1, loops):
        frames_cpy = np.concatenate(
            (
                frames_cpy,
                frames + l * (frames[-1] - frames[0] + first_frame_dur),
            )
        )
    return frames_cpy


class Presenter:
    """
    This class is responsible for presenting the stimuli. It is a wrapper around the
    pyglet window class and the moderngl_window BaseWindow class. It is responsible for
    loading the stimuli and presenting them. It is also responsible for
    communicating with the main process (gui) via a queue.
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
                    Whether to use fullscreen mode or not. Fullscreen is currently only
                    working on the main monitor.

        queue : multiprocessing.Queue
            Queue for communication with the main process (gui).

        """
        self.process_idx = process_idx
        self.config_dict = config_dict
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
        self.c_channels = config_dict["windows"][str(self.process_idx)][
            "channels"
        ]
        self.delay = delay
        self.arduino_running = False
        settings.WINDOW["class"] = (
            "moderngl_window.context.pyglet.Window"  # using a pyglet window
        )
        settings.WINDOW["gl_version"] = config_dict["gl_version"]
        settings.WINDOW["size"] = config_dict["windows"][str(self.process_idx)][
            "window_size"
        ]
        settings.WINDOW["aspect_ratio"] = (
            None  # Sets the aspect ratio to the window's aspect ratio
        )
        settings.WINDOW["fullscreen"] = config_dict["windows"][
            str(self.process_idx)
        ]["fullscreen"]
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True
        settings.WINDOW["resizable"] = False
        settings.WINDOW["title"] = "Noise Presentation"
        settings.WINDOW["style"] = config_dict["windows"][
            str(self.process_idx)
        ]["style"]

        self.frame_duration = (
            1 / config_dict["fps"]
        )  # Calculate the frame duration

        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (
            config_dict["windows"][str(self.process_idx)]["x_shift"],
            config_dict["windows"][str(self.process_idx)]["y_shift"],
        )  # Shift the window
        self.window.init_mgl_context()  # Initialize the moderngl context
        self.stop = False  # Flag for stopping the presentation
        self.window.set_default_viewport()  # Set the viewport to the window size

        if self.mode == "lead" and not config_dict["windows"][str(self.process_idx)]["arduino_port"] == "dummy":
            self.arduino = fpspy.arduino.Arduino(
                port=fpspy.config.get_arduino_port(config_dict),
                baud_rate=fpspy.config.get_arduino_baud_rate(config_dict),
                trigger_command=fpspy.config.get_arduino_trigger_command(
                    config_dict
                ),
                queue=ard_queue,
                queue_lock=ard_lock,
            )
        else:
            self.arduino = DummyArduino(
                port="COM_TEST",
                baud_rate=9600,
                queue=ard_queue,
                queue_lock=ard_lock,

            )


    def __del__(self):
        try:

            ard_lock = getattr(self, "ard_lock", None)
            arduino = getattr(self, "arduino", None)
            if ard_lock is not None and arduino is not None:
                try:
                    with ard_lock:
                        arduino.disconnect()
                except AttributeError:
                    pass
        except AttributeError:
            pass

    def run_empty(self):
        """
        Empty loop. Establishes a window filled with a grey background. Waits for
        commands from the main process (gui).
        """

        # if self.mode == "lead":
        #     self.arduino.send("W")
        while not self.window.is_closing:
            self.window.use()
            # self.window.ctx.clear(0.5, 0.5, 0.5, 1.0)  # Clear the window with a grey background
            self.window.ctx.clear(1, 1, 1, 1.0)

            self.window.swap_buffers()  # Swap the buffers (update the window content)
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.window.close()  # Close the window in case it is closed by the user

    def communicate(self):
        """Check and execute commands from the main process (gui)."""
        with self.lock:
            if self.queue.empty():
                return
            command = fpspy.queue.get(self.queue)

        match command.type:
            case "play":
                self.stop = False
                self.play(*command.args, **command.kwargs)
            case "white_screen":
                self.stop = False
                with self.ard_lock:
                    ard_command = self.ard_queue.get()
                    self.send_colour(ard_command)
                    self.arduino_running = True
                    arduino_thread = threading.Thread(
                        target=self.receive_arduino_status
                    )
                    arduino_thread.start()
            case "stop":  # If the command is "stop", stop the presentation
                self.arduino_running = False  # Trigger the stop flag for next time
                if self.mode == "lead":
                    self.status_queue.put("done")
                self.send_colour("b")
                self.send_colour("O")
                self.stop = True
                current_time = time.perf_counter()
                while time.perf_counter() - current_time < 1:
                    pass
            case "destroy":
                self.window.close()  # Close the window

    def receive_arduino_status(self):
        buffer = True
        if self.mode == "lead":
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
            self.arduino.send_trigger()

    def switch_trigger_modes(self, mode="t_s_off"):
        """Switch the trigger mode of the Arduino."""
        try:
            self.arduino.send(mode)
            self.arduino.arduino.flush()
        except AttributeError:
            pass

    def send_colour(self, colour):
        """Send a colour signal to the Arduino."""

        self.arduino.send(colour)

    def process_arduino_colours(
        self, colours: str, change_logic: int, n_frames: int
    ):
        """
        Processes the colours and calculates the necessary repeats.

        Parameters
        ----------
        colours : str
            A comma-separated string of colour values.
        change_logic : int
            The logic determining how often the colour changes.
        frames : int
            The total number of frames for the stimulus.

        Returns
        -------
        list
            A list of colours repeated and arranged as per the specified logic.
        """
        colours = colours.split(",")
        colour_repeats = np.ceil(n_frames / (len(colours) * change_logic))
        colours = np.repeat(np.asarray(colours), change_logic).tolist()
        colours = colours * int(colour_repeats)
        return colours

    def to_textures(self, stim: fpspy.Stim):
        """
        Create textures from the stimulus frames.

        Returns
        -------
        tuple
            A tuple containing the loaded patterns as a 3D array, the width and height
            of each pattern, the number of frames, and the desired frames per second
            (fps).
        """
        # Establish the texture for each stimulus frame
        textures = [
            self.window.ctx.texture(
                (stim.width, stim.height),
                stim.n_channels,
                stim.frames[i, :].tobytes(),
                samples=0,
                alignment=1,
            )
            for i in range(len(stim.frames))
        ]
        return textures

    def setup_shader_program(self, nr_colours=1):
        """
        Initializes the shader program using vertex and fragment shaders.

        Returns
        -------
        moderngl.Program
            The compiled and linked shader program.
        """
        # Load and compile vertex and fragment shaders
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / "vertex_shader.glsl").open("r") as vertex_file:
            vertex_shader_source = vertex_file.read()
        if nr_colours == 1:
            with (resource_dir / "fragment_shader.glsl").open(
                "r"
            ) as fragment_file:
                fragment_shader_source = fragment_file.read()
        else:
            with (resource_dir / "fragment_shader_colour.glsl").open(
                "r"
            ) as fragment_file:
                fragment_shader_source = fragment_file.read()

        # Create and return the shader program
        program = self.window.ctx.program(
            vertex_shader=vertex_shader_source,
            fragment_shader=fragment_shader_source,
        )

        return program

    def calculate_scaling(self, width, height):
        """Calculates scaling factors from texture and window aspect ratios.

        Parameters
        ----------
        width : int
            The width of the texture.
        height : int
            The height of the texture.

        Returns
        -------
        tuple
            A tuple of scaling factors (scale_x, scale_y) and the quad vertices array.
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
            The number of frames in the stimulus pattern.
        loops : int
            The number of times the stimulus pattern should loop.
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
        Main loop for presenting the stimulus.

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
            List of textures for each stimulus frame.
        program : moderngl.Program
            The shader program used for rendering.
        vao : moderngl.VertexArray
            The vertex array object for rendering.
        """

        if change_logic==1:
            self.window.ctx.clear(0, 0, 0)
            c = arduino_colours[0]

            self.send_colour(c)

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

            #Handle colour change logic
            if change_logic >1:
                if current_pattern_index % change_logic == 0:
                    c = arduino_colours[current_pattern_index]

                    self.send_colour(c)  # Custom function to send colour to Arduino

            # Clear the window and render the stimulus
            self.window.ctx.clear(0, 0, 0)
            patterns[current_pattern_index].use(location=0)
            if nr_colours > 1:
                program["pattern"].red = (
                    0  # Assuming 'pattern' is the uniform name in shader
                )
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
        return None

    def cleanup_and_finalize(
        self,
        patterns,
        vbo,
        vao,
        stimfile,
        loops,
        colours,
        change_logic,
        end_times,
        desired_fps,
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
        stim_dict : dict
            The dictionary containing stimulus settings, used for logging purposes.
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

            # Write log with the stim_dict or any other relevant information
        write_log(
            stimfile,
            loops,
            colours,
            change_logic,
            dropped_frames,
            wrong_frame_times,
        )

        # Run any additional emptying or resetting procedures
        self.stop = False
        self.arduino_running = False  # Trigger the stop flag for next time
        # if self.mode == "lead":
        #     with self.status_lock:
        #         self.status_queue.put("done")

        return

    def play(self, stim_path, loops, arduino_colours, change_logic, s_frames):
        """Play the stimulus file.

        This function loads the stimulus file, creates a texture from it and presents it.

        Parameters
        ----------
            - stim_path : str
                Path to the stimulus file.
            - loops
            - argduino_colours
            - change_logic
            - s_frames

        """
        s_frames = _loop(s_frames, loops)

        arduino_colours = self.process_arduino_colours(
            arduino_colours, change_logic, len(s_frames) - 1
        )

        # Load the stim data
        stim = fpspy.Stim.read_hdf5(stim_path)
        supports_channel_selection = stim.n_channels > 1
        if supports_channel_selection:
            stim = stim.with_channels(self.c_channels)
        textures = self.to_textures(stim)

        # Establish the shader program for presenting the stimulus.
        program = self.setup_shader_program(stim.n_channels)

        # Calculate scaling based on aspect ratio of window and stimulus.
        scale_x, scale_y, quad = self.calculate_scaling(stim.width, stim.height)

        # Create the buffer and vertex array object for the stimulus.
        vbo, vao = self.create_buffer_and_vao(quad, program)

        # Establish the time per frame for the desired fps

        time_per_frame, pattern_indices = self.setup_presentation(
            len(stim), loops, stim.fps
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
            f"stimulus will start in {s_frames[0] - time.perf_counter()} seconds, "
            f"window_idx: {self.process_idx}"
        )
        print(f"Current time is {datetime.datetime.now()}")
        end_times = np.zeros(len(s_frames))
        # Start the presentation loop
        self.switch_trigger_modes("t_s_on")
        end_times = self.presentation_loop(
            pattern_indices,
            s_frames,
            end_times,
            stim.n_channels,
            arduino_colours,
            change_logic,
            textures,
            program,
            vao,
        )
        self.switch_trigger_modes("t_s_off")

        # Clean up and finalize the presentation
        self.cleanup_and_finalize(
            textures,
            vbo,
            vao,
            stim_path,
            loops,
            arduino_colours,
            change_logic,
            end_times,
            stim.fps,
        )


def write_log(
    stimfile,
    loops,
    colours,
    change_logic,
    dropped_frames=None,
    wrong_frame_times=None,
):
    """
    Write the log file for the stimulus presentation.
    Parameters
    ----------
    stim_dict : str
        Path to the stimulus file.
    """
    filename_format = (
        fpspy.config.user_log_dir()
        / f"{stimfile}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')}"
    )

    if dropped_frames is None:
        dropped_frames = []
        wrong_frame_times = []

    with open(filename_format, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stim_file",
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
                stimfile,
                loops,
                colours,
                change_logic,
                time.strftime("%H:%M:%S"),
                dropped_frames,
                wrong_frame_times,
            ]
        )


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
    Start the pyglet app.

    This function spawns the pyglet app in a separate process.

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
    presenter = Presenter(
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
    presenter.run_empty()  # Establish the empty loop


def pyglet_app_follow(
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
    Start the pyglet app.

    This function is spawns the pyglet app in a separate process.

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
    presenter = Presenter(
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
        mode="follow",
        delay=delay,
    )
    presenter.run_empty()  # Establish the empty loop


# Can run the pyglet app from here for testing purposes if needed
# if __name__ == '__main__':
#     pyglet_app(Queue())
