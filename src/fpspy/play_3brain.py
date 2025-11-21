"""3Brain presenters.

Differences to play.py:

  - Slimmed down presenter class that doesn't handle Arduino communication.
  - Has support for broadcasting stimuli (1x1 textures -> full screen color).
  - Exposes callbacks to allow for external code to hook into events like buffer swap.
"""

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


def _loop(s_frames, loops):
    """Loop the frames a given number of times.

    Returns
    -------
    frame_idxs : np.ndarray
        Frame indices for the looped frames.
    s_frames : np.ndarray
        Frame schedule for the looped frames.
    """
    # Frame indices.
    idxs = np.tile(np.arange(len(s_frames)), loops)
    # Frame schedule.
    s_frames_cpy = s_frames.copy()
    first_frame_dur = s_frames[1] - s_frames[0]
    for l in range(1, loops):
        s_frames_cpy = np.concatenate(
            (
                s_frames_cpy,
                s_frames + l * (s_frames[-1] - s_frames[0] + first_frame_dur),
            )
        )
    assert len(idxs) == len(s_frames_cpy)
    return idxs, s_frames_cpy


def _delay(s_frames, delay):
    """Add a delay to the frame schedule.

    The delay must be such that the first frame is still in the future.

    Returns
    -------
    s_frames : np.ndarray
        Frame schedule with added delay.
    """
    s_frames = s_frames + delay
    if s_frames[0] <= time.perf_counter():
        raise Exception("Failed to start stimulus in time. Increased delay needed.")
    return s_frames


def _schedule_frames(t0, frames, fps):
    """Schedule frames starting from time t0.

    Returns
    -------
    s_frames : np.ndarray
        Frame schedule.
    """
    frame_duration = 1 / fps
    s_frames = __import__("numpy").linspace(
        t0, t0 + frames * frame_duration, frames + 1
    )
    start_times = s_frames[:-1]
    return start_times


def _wait_until(target_time):
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
        self.status_queue = status_queue
        self.c_channels = config["windows"][str(self.process_idx)]["channels"]
        self.delay = delay

        # Create a logger adapter that automatically includes window_idx
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__), {"window_idx": self.process_idx}
        )
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
        settings.WINDOW["fullscreen"] = config["windows"][str(self.process_idx)][
            "fullscreen"
        ]
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True
        settings.WINDOW["resizable"] = False
        settings.WINDOW["title"] = "Noise Presentation"
        settings.WINDOW["style"] = config["windows"][str(self.process_idx)]["style"]

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
        A list of moderngl.Texture objects.
        """
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

    def setup_shader_program(self):
        """
        Initializes the shader program using vertex and fragment shaders.

        Returns
        -------
        moderngl.Program
            The compiled and linked shader program.
        """
        # Load shaders.
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / "vertex_shader.glsl").open("r") as vertex_file:
            vertex_shader_source = vertex_file.read()
        with (resource_dir / "fragment_shader_colour.glsl").open("r") as fragment_file:
            fragment_shader_source = fragment_file.read()
        # Compile and link the shader program.
        program = self.window.ctx.program(
            vertex_shader=vertex_shader_source,
            fragment_shader=fragment_shader_source,
        )
        # We always use 1 texture, bound to texture unit 0.
        program["tex"].value = self.TEXTURE_UNIT
        return program

    @staticmethod
    def create_quad(stim_width, stim_height, win_width, win_height):
        """Create a quad that has corners at each stimulus corner.

        If the stimulus is 1x1 pixel, the quad covers the whole screen (broadcasting).

        Parameters
        ----------
        stim_width : int
            The width of the stimulus texture in pixels.
        stim_height : int
            The height of the stimulus texture in pixels.
        width : int
            The width of the window in pixels.
        height : int
            The height of the window in pixels.

        Returns
        -------
        tuple
            A tuple of scaling factors (scale_x, scale_y) and the quad vertices array.
        """
        # Calculate the aspect ratio of the window and the texture
        window_aspect = win_width / win_height
        texture_aspect = stim_width / stim_height
        # If the stimulus has shape (f, h, w, c) == (f, 1, 1, c), we broadcast by
        # having the single pixel cover the whole screen.
        do_broadcast = stim_height == 1 and stim_width == 1
        if do_broadcast:
            scale_x = scale_y = 1
        else:
            # Determine scaling factors based on aspect ratios
            scale_x = stim_width / win_width
            scale_y = stim_height / win_height
            # TODO: what is this for?
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
        return quad

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

    def presentation_loop(self, texture_idxs, s_frames, textures, vao):
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
        end_times = np.zeros(len(s_frames))
        for i, texture_idx in enumerate(texture_idxs):
            is_exit = self.communicate()
            if is_exit:
                return end_times
            # Sync frame presentation to the scheduled time.
            _wait_until(s_frames[i])

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
        end_times,
        desired_fps,
    ):
        """Clean up resources and write logs.

        Parameters
        ----------
        patterns : list
            List of texture objects to be released.
        vbo : moderngl.Buffer
            The vertex buffer object to be released.
        vao : moderngl.VertexArray
            The vertex array object to be released.
        stimfile : str
            Path to the stimulus file.
        loops : int
            Number of loops the stimulus was presented.
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
            dropped_frames,
            wrong_frame_times,
        )
        return

    def play(self, stim_path, loops, t0):
        """Play the stimulus file.

        Loads the stimulus file, creates textures from it and presents them.

        Parameters
        ----------
            - stim_path : str
                Path to the stimulus file.
            - loops
                Number of times to loop the stimulus.
            - s_frames
                The frame schedule: list of times to show each frame.
        """
        # Load the stim data
        stim = fpspy.Stim.read_hdf5(stim_path)
        s_frames = _schedule_frames(t0, len(stim), stim.fps)
        if len(stim) != len(s_frames):
            raise ValueError(f"{len(stim)=} ≠ {len(s_frames)=}.")
        frame_idxs, s_frames = _loop(s_frames, loops)
        supports_channel_selection = stim.n_channels > 1
        if supports_channel_selection:
            stim = stim.with_channels(self.c_channels)
        textures = self.to_textures(stim)

        # Establish the shader program for presenting the stimulus.
        program = self.setup_shader_program()
        # Calculate scaling based on aspect ratio of window and stimulus.
        quad = self.create_quad(stim.width, stim.height, *self.window.size)
        # Create the buffer and vertex array object for the stimulus.
        vbo, vao = self.create_buffer_and_vao(quad, program)

        # Add delay to frames.
        s_frames = _delay(s_frames, self.delay)

        self.logger.info(
            f"stimulus will start in {s_frames[0] - time.perf_counter()} seconds"
        )
        self.logger.info(f"Current time is {datetime.datetime.now()}")

        # Start the presentation loop
        end_times = self.presentation_loop(frame_idxs, s_frames, textures, vao)
        # Clean up and finalize the presentation
        self.cleanup_and_finalize(
            textures, vbo, vao, stim_path, loops, end_times, stim.fps
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
    logfile = (
        fpspy.config.user_log_dir() / f"{stimfile.stem}_"
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')}"
    )

    if dropped_frames is None:
        dropped_frames = []
        wrong_frame_times = []

    with open(logfile, "a", newline="") as f:
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


def pyglet_app(
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
    # Configure logging for this child process
    # Each process needs its own logging configuration
    fpspy._logging.setup_logging(log_level)

    presenter = Presenter(
        process_idx,
        config,
        queue,
        status_queue,
        delay=delay,
    )
    presenter.run_empty()
