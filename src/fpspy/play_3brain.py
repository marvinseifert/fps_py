import csv
import datetime
import importlib.resources
import logging
import time
from typing import Callable
import moderngl
import moderngl_window
from moderngl_window.conf import settings
import numpy as np
import fpspy.config
import fpspy.queue
import fpspy._logging


_logger = logging.getLogger(__name__)


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


def wait_until(target_time):
    """Semi-busy-wait until the target time is reached."""
    max_busy_wait_ms = 0.002
    now = time.perf_counter()
    remaining = target_time - now
    if remaining <= 0:
        _logger.warning(f"{target_time=} already passed, {now=}.")
        return
    if remaining > max_busy_wait_ms:
        time.sleep(remaining - max_busy_wait_ms)
    while time.perf_counter() < target_time:
        pass


# Callback is given the frame index.
BufferSwapCallback = Callable[[int], None]


class Presenter:
    """3Brain presenter.

    Slimmed-down version of play.Presenter (which is used for the Multi-Channel Systems
    setup).

    This class is a wrapper around the pyglet window class and the moderngl_window
    BaseWindow class. It is responsible for loading and presenting the stimuli.
    The Presenter class is instantiated in a separate process for each window
    and communicate with the GUI process via a multiprocessing.Queue.
    """

    def __init__(
        self,
        process_idx,
        config,
        queue,
        status_queue,
        mode,
        delay=10,
    ):
        """
        Parameters
        ----------
        config : dict
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
        self.config = config
        self.queue = queue
        self.mode = mode
        self.status_queue = status_queue
        self.c_channels = config["windows"][str(self.process_idx)]["channels"]
        self.delay = delay
        # Create a logger adapter that automatically includes window_idx
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__), {"window_idx": self.process_idx}
        settings.WINDOW["class"] = (
            "moderngl_window.context.pyglet.Window"  # using a pyglet window
        )
        settings.WINDOW["gl_version"] = config["gl_version"]
        settings.WINDOW["size"] = config["windows"][str(self.process_idx)][
            "window_size"
        ]
        settings.WINDOW["aspect_ratio"] = (
            None  # Sets the aspect ratio to the window's aspect ratio
        )
        settings.WINDOW["fullscreen"] = config["windows"][
            str(self.process_idx)
        ]["fullscreen"]
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True
        settings.WINDOW["resizable"] = False
        settings.WINDOW["title"] = "Noise Presentation"
        settings.WINDOW["style"] = config["windows"][str(self.process_idx)][
            "style"
        ]

        self.frame_duration = 1 / config["fps"]  # Calculate the frame duration

        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (
            config["windows"][str(self.process_idx)]["x_shift"],
            config["windows"][str(self.process_idx)]["y_shift"],
        )  # Shift the window
        self.window.init_mgl_context()  # Initialize the moderngl context
        self.window.set_default_viewport()  # Set the viewport to the window size

        # We only use 1 texture unit, so we can hardcode its unit number.
        self.TEXTURE_UNIT = 0

        # Callbacks. Currently only allows for one callback per event.
        # Purpose: to allow for arduino color changing.
        self.on_swap_buffers_before = None
        self.on_swap_buffers_after = None
        self.on_stop = None

    def register_on_swap_buffers_before(self, callback: BufferSwapCallback):
        """Register a callback for the before swap buffers event."""
        self.on_swap_buffers_before = callback

    def register_on_swap_buffers_after(self, callback: BufferSwapCallback):
        """Register a callback for the after swap buffers event."""
        self.on_swap_buffers_after = callback

    def register_on_stop(self, callback: Callable[[], None]):
        """Register a callback for the stop event."""
        self.on_stop = callback

    def notify_swap_buffers_before(self, frame_idx: int):
        """Notify the before swap buffers event."""
        if self.on_swap_buffers_before is not None:
            self.on_swap_buffers_before(frame_idx)

    def notify_swap_buffers_after(self, frame_idx: int):
        """Notify the after swap buffers event."""
        if self.on_swap_buffers_after is not None:
            self.on_swap_buffers_after(frame_idx)

    def notify_stop(self):
        """Notify the stop event."""
        if self.on_stop is not None:
            self.on_stop()

    def run_empty(self):
        """
        Empty loop. Establishes a window filled with a grey background. Waits for
        commands from the main process (gui).
        """
        self.window.use()
        while not self.window.is_closing:
            # Clear the window with a grey background
            # self.window.ctx.clear(0.5, 0.5, 0.5, 1.0)
            self.window.ctx.clear(0, 0, 0, 1.0)
            self.window.swap_buffers()  # Swap the buffers (update the window content)
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.window.close()  # Close the window in case it is closed by the user

    def communicate(self):
        """Check and execute commands from the main process (gui)."""
        if self.queue.empty():
            return
        command = fpspy.queue.get(self.queue)

        stop = False
        match command.type:
            case "play":
                self.play(*command.args, **command.kwargs)
            case "white_screen":
                pass
            case "stop":
                self.notify_stop()
                self.status_queue.put("done")
                stop = True
                current_time = time.perf_counter()
                # Why wait 1 second here?
                while time.perf_counter() - current_time < 1:
                    pass
            case "destroy":
                stop = True
                self.window.close()  # Close the window
        return stop

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
                stim.frames[i].tobytes(),
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
        # We always use 1 texture, bound to texture unit 0.
        program["tex"].value = self.TEXTURE_UNIT
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
        texture_idxs,
        s_frames,
        end_times,
        textures,
        vao,
    ):
        """
        Main loop for presenting the stimulus.

        Parameters
        ----------
        texture_idxs : list
            Indices indicating the order in which to present the textures.
        s_frames : list
            Timestamps for when each frame should start.
        textures : list
            Textures for each stimulus frame.
        vao : moderngl.VertexArray
            The vertex array object for rendering.
        """
        # Ensure the correct context is being used.
        self.window.use()
        for i, texture_idx in enumerate(texture_idxs):
            is_exit = self.communicate()
            if is_exit:
                return end_times
            # Sync frame presentation to the scheduled time.
            wait_until(s_frames[i])

            # Clear the window and render the stimulus.
            self.window.ctx.clear(0, 0, 0)
            textures[texture_idx].use(location=self.TEXTURE_UNIT)
            vao.render(moderngl.TRIANGLES)

            # Swap buffers, wrapped by callbacks and time record.
            self.notify_swap_buffers_before(i)
            start_time = time.perf_counter()
            self.window.swap_buffers()
            end_times[i] = time.perf_counter() - start_time
            self.notify_swap_buffers_after(i)
        return end_times

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
            self.logger.error(f"dropped frames (idx): {dropped_frames[0]}")
            self.logger.error(f"wrong frame times: {wrong_frame_times}")

            # Write log with the stim_dict or any other relevant information
        write_log(
            stimfile,
            loops,
            colours,
            change_logic,
            dropped_frames,
            wrong_frame_times,
        )
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

        # Load the stim data
        stim = fpspy.Stim.read_hdf5(stim_path)
        supports_channel_selection = stim.n_channels > 1
        if supports_channel_selection:
            stim = stim.with_channels(self.c_channels)
        textures = self.to_textures(stim)

        # Establish the shader program for presenting the stimulus.
        program = self.setup_shader_program(stim.n_channels)

        # Calculate scaling based on aspect ratio of window and stimulus.
        do_broadcast = stim.height == 1 and stim.width == 1
        if do_broadcast:
            _, _, quad = self.calculate_scaling(*self.window.size)
        else:
            _, _, quad = self.calculate_scaling(stim.width, stim.height)

        # Create the buffer and vertex array object for the stimulus.
        vbo, vao = self.create_buffer_and_vao(quad, program)

        # Establish the time per frame for the desired fps

        time_per_frame, pattern_indices = self.setup_presentation(
            len(stim), loops, stim.fps
        self.logger.info(
            f"stimulus will start in {s_frames[0] - time.perf_counter()} seconds"
        )

        # Synchronize the presentation

        # Add buffer delay to frames:
        delay_needed = s_frames[0] - time.perf_counter()
        if delay_needed > 0:
            delay = 10
        else:
            delay = np.abs(delay_needed) + 10
        s_frames = s_frames + delay

        end_times = np.zeros(len(s_frames))
        self.logger.info(f"Current time is {datetime.datetime.now()}")

        # Start the presentation loop
        end_times = self.presentation_loop(
            pattern_indices,
            s_frames,
            end_times,
            textures,
            vao,
        )

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
    status_queue,
    delay=10,
    log_level="INFO",
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
    log_level : str
        Logging level for this process (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    presenter = Presenter(
        process_idx,
        config,
        queue,
        status_queue,
        mode="lead",
        delay=delay,
    )
    presenter.run_empty()

    # Configure logging for this child process
    # Each process needs its own logging configuration
    fpspy._logging.setup_logging(log_level)

def pyglet_app_follow(
    process_idx,
    config,
    queue,
    status_queue,
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
        status_queue,
        mode="follow",
        delay=delay,
    )
    presenter.run_empty()
