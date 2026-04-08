"""3Brain presenters.

Differences to play.py:

  - Slimmed down presenter class that doesn't handle Arduino communication.
  - Has support for broadcasting stimuli (1x1 textures -> full screen color).
  - Exposes callbacks to allow for external code to hook into events like buffer swap.
"""

import csv
import datetime
import logging
from pathlib import Path
import time
from typing import Callable, Optional, Literal
import multiprocessing as mp
import moderngl
import moderngl_window
from moderngl_window.conf import settings
import numpy as np
import fpspy.config
import fpspy.queue
import fpspy._logging
import fpspy.arduino
import OpenGL.GL as gl
import fpspy.stim as stim
import fpspy.color
import dataclasses

# We are okay with any interleaving caused by multiple processes. The logs are not
# critical, and each process has its own log files anyway.
_logger = logging.getLogger(__name__)

RenderTarget = Literal["screen", "array"]


def probe_default_fbo_srgb() -> tuple[bool, bool]:
    """Check if the default framebuffer has sRGB enabled.

    ModernGL is opaque about whether the default framebuffer is sRGB-capable, so we
    check manually.
    """
    # Are we converting linear->sRGB on framebuffer writes?
    srgb_enabled = bool(gl.glIsEnabled(gl.GL_FRAMEBUFFER_SRGB))

    # Is the default framebuffer attachment sRGB-encoded or linear?
    enc = gl.glGetFramebufferAttachmentParameteriv(
        gl.GL_FRAMEBUFFER,
        gl.GL_BACK_LEFT,  # default framebuffer color buffer
        gl.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
    )
    # enc is an int enum: GL_SRGB or GL_LINEAR
    srgb_capable = int(enc) == int(gl.GL_SRGB)
    return srgb_capable, srgb_enabled


def _wait_or_skip(target_time, next_frame_time: Optional[float], fps):
    """Semi-busy-wait for the frame time, or possibly skip to next frame."""
    max_busy_wait_ms = 0.002
    now = time.perf_counter()
    remaining = target_time - now
    if remaining <= 0:
        _logger.warning(f"{target_time=} already passed, {now=}.")
        has_next_frame = next_frame_time is not None
        if not has_next_frame:
            return False
        half_period = 0.5 / fps
        # If displaying the current frame would cause us to miss the next frame by more
        # than half a period, then skip to the next frame.
        if now + half_period >= next_frame_time:
            _logger.warning("Skipping to next frame.")
            return True
        return False
    if remaining > max_busy_wait_ms:
        time.sleep(remaining - max_busy_wait_ms)
    while time.perf_counter() < target_time:
        pass
    return False


# Callback is given the frame index.
OnTriggerCallback = Callable[[], None]


@dataclasses.dataclass
class PlayState:
    """State needed to be stored when stepping through a stimulus via commands."""

    prog: stim.StimProgram
    frame_idxs: np.ndarray
    triggers: np.ndarray
    current_frame: int


class Presenter:
    """3Brain presenter.

    Slimmed-down version of play.Presenter (which is used for the Multi-Channel Systems
    setup).

    This class is a wrapper around the pyglet window class and the moderngl_window
    BaseWindow class. It is responsible for loading and presenting the stimuli.
    The Presenter class is instantiated in a separate process for each window
    and communicate with the GUI process via a multiprocessing.Queue.
    """

    dropped_frames_filename_fmt = "dropped_frames_win{window_idx}.csv"

    def __init__(
        self,
        process_idx,
        config,
        out_dir,
        cmd_queue,
        status_queue,
        delay=10,
    ):
        """
        Parameters
        ----------
        config : dict
            See notes below.
        out_dir : str or Path
            Output directory for data such as dropped frames.
        cmd_queue : multiprocessing.Queue
            Queue for receiving commands from the main process.
        status_queue : multiprocessing.Queue
            Queue for sending status updates to the main process.
        delay : float
            Delay added before starting the stimulus presentation, in seconds.

        config
        ------
        Dictionary containing the configuration parameters for one or more windows.

        An example showing the expected structure:
        {
            "gl_version": [4, 1],
            "fps" 75.0,
            "windows": {
                "1": {
                    "x_shift": 0,
                    "y_shift": 0,
                    "window_size": [800, 600],
                    "fullscreen": False,
                    "style": "transparent",
                    "channels": [1, 2, 3],
                },
                "2": { ... },
        }

        Keys:
            "gl_version" : tuple
                Version of OpenGL to use.
                Size of the window.
            "fps" : float
                Frames per second for the stimulus presentation.
            "windows" : dict
                Dictionary containing the window-specific parameters. Each key is
                the process index (as string) and the value is another dictionary.
        Each window dictionary:
            "y_shift" : int
                Shift of the window in y direction.
            "x_shift" : int
                Shift of the window in x direction.
            "window_size" : tuple
                Size of the window as (width, height).
            "fullscreen" : bool
                Whether to use fullscreen mode or not. Fullscreen is currently only
                working on the main monitor.
            "style" : str
                Style of the window.
            "channels" : list of int
                List of channels to present on this window.
            "clear_rgba" : list of float
                Clear color for the window as [r, g, b, a].

        """
        self.process_idx = process_idx
        self.config = config
        self.queue = cmd_queue
        self.status_queue = status_queue
        self.out_dir = Path(out_dir)
        self.delay = delay
        self.setup_logging()
        self.setup_window(config)

        # Callbacks. Currently only allows for one callback per event.
        # Purpose: to allow for arduino color changing.
        self._on_trigger = None
        self._on_stop = None
        # Play state is used when loading and stepping (not needed for one-shot play).
        self.play_state: Optional[PlayState] = None

    def _is_step_play(self):
        """Check if we are in step-play mode."""
        return self.play_state is not None and self.play_state.current_frame >= 0

    def setup_window(self, config):
        settings.WINDOW["class"] = "moderngl_window.context.pyglet.Window"
        settings.WINDOW["gl_version"] = config["gl_version"]
        window_config = config["windows"][str(self.process_idx)]
        settings.WINDOW["size"] = window_config["window_size"]
        settings.WINDOW["fullscreen"] = window_config["fullscreen"]
        settings.WINDOW["style"] = window_config["style"]
        settings.WINDOW["aspect_ratio"] = None
        settings.WINDOW["samples"] = 0
        settings.WINDOW["double_buffer"] = True
        settings.WINDOW["vsync"] = True
        settings.WINDOW["resizable"] = False
        settings.WINDOW["title"] = f"Win {self.process_idx} Presenter"

        self.c_channels = window_config["channels"]
        self.mirror = window_config["mirror"]
        self.rotation = window_config["rotation"]
        # Assume constant. Keep both fps and duration for convenience.
        self.fps = config["fps"]
        self.frame_duration = 1 / self.fps
        self.clear_rgba = window_config["clear_rgba"]
        if self.clear_rgba == [0.5, 0.5, 0.5, 1.0]:
            self.logger.warning(
                "Received {self.clear_rgba} as clear color. RGB values "
                "are interpreted as sRGB, so 0.5  will map to ~0.22 in "
                "intensity when displayed. 50% grey in sRGB is ~0.73."
            )
        self.window = moderngl_window.create_window_from_settings()
        self.window.position = (window_config["x_shift"], window_config["y_shift"])
        self.window.init_mgl_context()
        self.window.set_default_viewport()

    def setup_logging(self):
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__), {"window_idx": self.process_idx}
        )
        fpspy._logging.enable_file_logging(
            self.out_dir / f"presenter_{self.process_idx}.log"
        )

    def close_window(self):
        """Close the window."""
        if self.window is not None:
            self.window.close()

    def register_on_trigger(self, callback: OnTriggerCallback):
        """Register a callback for the after swap buffers event."""
        self._on_trigger = callback

    def register_on_stop(self, callback: Callable[[], None]):
        """Register a callback for the stop event."""
        self._on_stop = callback

    def notify_stop(self):
        """Notify the stop event."""
        if self._on_stop is not None:
            self._on_stop()

    def notify_trigger(self):
        """Notify the trigger event."""
        if self._on_trigger is not None:
            self._on_trigger()

    def run_empty(self):
        """Do nothing, waiting for commands from the main process."""
        self.window.use()
        while not self.window.is_closing:
            # Only clear/swap if no stimulus is loaded (otherwise show_frame handles it)
            if not self._is_step_play():
                self.window.ctx.clear(*self.clear_rgba)
                self.window.swap_buffers()
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.close_window()

    def communicate(self):
        """Check and execute commands from the main process (gui)."""
        if self.queue.empty():
            return
        command = fpspy.queue.get_from(self.queue)

        do_stop = False
        do_destroy = False
        match command.type:
            case "white_screen":
                pass
            case "load":
                self.load(*command.args, **command.kwargs)
                # Send back total frame count for GUI (don't show frame yet)
                if self.play_state is not None:
                    total = len(self.play_state.frame_idxs)
                    self.status_queue.put({"total_frames": total})
            case "step_next":
                if self.play_state is not None:
                    do_stop = self.step_next(**command.kwargs)
                    # current_frame is now the frame that is displayed
                    frame_shown = self.play_state.current_frame if self.play_state else -1
                    self.status_queue.put({"stepped": frame_shown})
                else:
                    self.status_queue.put("no_stimulus")
            case "step_prev":
                if self.play_state is not None:
                    do_stop = self.step_prev(**command.kwargs)
                    # current_frame is now the frame that is displayed (-1 for cleared)
                    frame_shown = self.play_state.current_frame
                    self.status_queue.put({"stepped": frame_shown})
                else:
                    self.status_queue.put("no_stimulus")
            case "play":
                do_stop = self.play(*command.args, **command.kwargs)
            case "stop":
                do_stop = True
            case "destroy":
                do_destroy = True
        if do_stop or do_destroy:
            self.notify_stop()
            self.status_queue.put("done")
        if do_destroy:
            self.close_window()

    def _load(self, stim_path, stim_config, loops, t0, speed):
        """Load a stimuli; shared by load() and play()."""
        prog = stim.create_program(stim_path, stim_config)
        _logger.info(f"[start] program setup")
        s_frames, triggers = prog.setup(
            self.window.ctx, *self.window.size, self.c_channels,
            self.mirror, self.rotation, self.process_idx
        )
        _logger.info(f"[end] program setup")
        # The presenter can delay and loop a stimulus.
        s_frames = s_frames * speed + t0
        frame_idxs, s_frames, triggers = fpspy.stim.loop(s_frames, triggers, loops)
        triggers_arr = stim.decompress_triggers(triggers, len(frame_idxs))
        assert len(frame_idxs) == len(s_frames) - 1
        return prog, frame_idxs, s_frames, triggers_arr

    def load(self, stim_path, stim_config, loops):
        """Load (for step-play) one of the supported stimuli.

        Parameters
        ----------
        path : str or Path
            Path to the shader program file.
        stim_config : str or None
            Optional JSON config string for the stimulus.
        """
        prog, frame_idxs, s_frames, triggers_arr = self._load(
            stim_path, stim_config, loops, t0=0.0, speed=1.0
        )
        # We don't need s_frames here.
        self.play_state = PlayState(
            prog=prog,
            frame_idxs=frame_idxs,
            triggers=triggers_arr,
            current_frame=-1,
        )

    def step_next(self, close_if_done=False):
        """Step to the next frame.

        current_frame represents the frame currently displayed (-1 if none).
        """
        assert self.play_state is not None, "No stimulus loaded."
        state = self.play_state
        next_frame = state.current_frame + 1
        if next_frame >= len(state.frame_idxs):
            if close_if_done:
                state.prog.cleanup()
                self.play_state = None
            return close_if_done
        state.current_frame = next_frame
        self.show_frame(state.current_frame)
        return False

    def step_prev(self, close_if_done=False):
        """Step to previous frame, or clear screen if going to -1."""
        assert self.play_state is not None, "No stimulus loaded."
        state = self.play_state
        prev_frame = state.current_frame - 1
        state.current_frame = prev_frame
        if prev_frame < 0:
            self.clear_screen()
        else:
            self.show_frame(state.current_frame)
        return False

    def clear_screen(self):
        """Clear the screen to black."""
        self.window.use()
        self.window.ctx.clear(0, 0, 0)
        self.window.swap_buffers()

    def show_frame(self, frame):
        """Step one frame in the loaded stimulus.

        This function operates similar to shader_loop().

        Returns
        -------
        bool
            Whether to close the presenter after this step.
        """
        assert self.play_state is not None, "No stimulus loaded."
        state = self.play_state
        self.window.use()
        # Clear window (to black is fine), render the stimulus and swap buffers.
        self.window.ctx.clear(0, 0, 0)
        state.prog.render(self.window.ctx, state.frame_idxs[frame], frame)
        self.window.swap_buffers()
        if state.triggers[frame]:
            self.notify_trigger()

    def play(self, stim_path, stim_config, loops, t0, speed=None, close_after=False):
        """Play one of the supported stimuli.

        Loads the shader and metadata from files, then renders the stimulus
        procedurally on the GPU without loading frames into memory.

        Parameters
        ----------
        path : str or Path
            Path to the shader program file.
        loops : int
            Number of times to loop the stimulus.
        t0 : float
            Reference time (from time.perf_counter()).
        speed : float, optional
            Override the playback speed.
        """
        speed = speed if speed is not None else 1.0
        prog, frame_idxs, s_frames, triggers_arr = self._load(
            stim_path, stim_config, loops, t0, speed
        )
        s_frames = stim.delay(s_frames, self.delay)
        self.logger.info(f"Starting in {s_frames[0] - time.perf_counter():.3f} s.")
        dropped_frames = self.shader_loop(prog, frame_idxs, s_frames, triggers_arr)
        self.record_dropped_frames(dropped_frames)
        prog.cleanup()
        return close_after

    def shader_loop(self, prog: stim.StimProgram, frame_idxs, s_frames, triggers):
        """
        Main loop for presenting the stimulus.
        """
        assert len(frame_idxs) == len(s_frames) - 1
        N = len(frame_idxs)
        self.window.use()
        srgb_capable, srgb_enabled = probe_default_fbo_srgb()
        _logger.debug(
            f"Framebuffer sRGB capable: {srgb_capable}, enabled: {srgb_enabled}"
        )
        dropped_frames = []
        for i in range(N):
            is_exit = self.communicate()
            if is_exit:
                return dropped_frames

            # Sync frame presentation to the scheduled time.
            next_frame_time = s_frames[i + 1] if i < N - 1 else None
            skip = _wait_or_skip(s_frames[i], next_frame_time, self.fps)
            if skip:
                dropped_frames.append(i)
                continue

            # Clear window (to black is fine), render the stimulus and swap buffers.
            self.window.ctx.clear(0, 0, 0)
            prog.render(self.window.ctx, frame_idxs[i], i)
            self.window.swap_buffers()
            if triggers[i]:
                self.notify_trigger()
        return dropped_frames

    def record_dropped_frames(self, dropped_frames):
        """Record dropped frames to a CSV file."""
        if not dropped_frames:
            self.logger.info("No dropped frames.")
            return
        out_path = self.out_dir / self.dropped_frames_filename_fmt.format(
            window_idx=self.process_idx
        )
        with open(out_path, "w", newline="") as f:
            res = np.array(dropped_frames, copy=False)
            np.savetxt(f, res, fmt="%d", delimiter=",")
        self.logger.warning(f"Dropped frames: {dropped_frames}\t(saved to {out_path})")


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
        fpspy.config.default_log_dir() / f"{stimfile.stem}_"
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
    out_dir,
    cmd_queue,
    status_queue,
    delay,
    enable_triggers,
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
    out_dir : str or Path
        Output directory for logs and output data.
    cmd_queue : multiprocessing.Queue
        Queue for receiving commands from the main process.
    status_queue : multiprocessing.Queue
        Queue for sending status updates to the main process.
    delay : float
        Delay added before starting the stimulus presentation, in seconds.
    render_target : RenderTarget
        Render target for the stimulus presentation ('screen' or 'array').
    enable_triggers : bool
        Whether to enable Arduino triggers during the presentation.
    log_level : str
        Logging level for this process (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    presenter = Presenter(
        process_idx,
        config,
        out_dir,
        cmd_queue,
        status_queue,
        delay=delay,
    )
    if enable_triggers:
        arduino = fpspy.arduino.Arduino(
            port=fpspy.config.get_arduino_port(config),
            baud_rate=fpspy.config.get_arduino_baud_rate(config),
            trigger_command=fpspy.config.get_arduino_trigger_command(config),
        )
        # Send arduino triggers on buffer swap
        presenter.register_on_trigger(arduino.send_trigger)
    presenter.run_empty()


class ArrayRenderer:
    """Render to an offscreen array.

    Useful reference: https://github.com/szabolcsdombi/headless-moderngl-experiment/blob/master/src/main_multisample.py

    Note: currently, we don't support multisampling, so there may be aliasing present
    that is not present when rendering to screen with multisampling enabled.

    Unlike Presenter, this class is assumed to be used in the main process—not
    spawned in a separate process. Because of this, a few things are different:

      - The program can be created once before being sent to render(), and so the
        arguments to render() are different, accepting a pre-created StimProgram.
      - Logging doesn't need to worry about multiprocessing, so the file's _logger can
        be used, instead of per-process loggers.
    """

    def __init__(self, process_idx, config):
        self.process_idx = process_idx
        window_config = config["windows"][str(self.process_idx)]
        self.c_channels = window_config["channels"]
        self.mirror = window_config["mirror"]
        self.rotation = window_config["rotation"]
        self.window_size = window_config["window_size"]  # (width, height)
        self.clear_rgba = window_config["clear_rgba"]
        self.ctx = moderngl.create_context(standalone=True)

    def close_ctx(self):
        """Close the context."""
        if self.ctx is not None:
            self.ctx.release()

    def __del__(self):
        self.close_ctx()

    def render(self, prog: stim.StimProgram, convert_to_srgb=False):
        """Render one of the supported shader-based stimuli to an array.

        Loads the shader and metadata from files, then renders the stimulus
        procedurally on the GPU without loading frames into memory.

        Parameters
        ----------
        stim_path : str or Path
            Path to the shader program file.
        stim_config : str or None
            Optional JSON config string for the stimulus.
        convert_to_srgb : bool
            Whether to convert the output from RGB to sRGB color space.
        """
        # The GUI can customize the stimulus through the config.
        s_frames, triggers = prog.setup(
            self.ctx, *self.window_size, self.c_channels,
            self.mirror, self.rotation, self.process_idx
        )
        n_frames = len(s_frames) - 1
        frame_idxs = np.arange(n_frames)
        arr = self.shader_loop(prog, frame_idxs)
        prog.cleanup()

        # Save frames
        if convert_to_srgb:
            arr = (fpspy.color.to_srgb(arr.astype(np.float32) / 255.0) * 255.0).astype(
                np.uint8
            )
        _logger.info(
            f"Rendered {len(frame_idxs)} frames to array, with {len(triggers)} "
            f"triggers, for window {self.process_idx}."
        )
        return arr, s_frames, triggers

    def shader_loop(self, shader: stim.StimProgram, frame_idxs):
        """
        Main loop for rendering the stimulus to numpy arrays.

        Parameters
        ----------
        shader : stim.StimProgram
            The shader program to render.
        frame_idxs : array-like
            Array of frame indices to render.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, height, width, 3) with rendered frames.
        """
        width, height = self.window_size
        n_frames = len(frame_idxs)

        # Create FBO with a texture for offscreen rendering
        fbo_texture = self.ctx.texture((width, height), 4)  # RGBA
        fbo = self.ctx.framebuffer(color_attachments=[fbo_texture])

        # Pre-allocate output array (RGB only, drop alpha)
        frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
        fbo.use()
        for i, frame_idx in enumerate(frame_idxs):
            self.ctx.clear(*self.clear_rgba)
            shader.render(self.ctx, frame_idx, i)
            # Read pixels from the FBO
            data = fbo_texture.read()
            frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            # Keep RGB channels only
            frames[i] = frame[:, :, :3]
        fbo.release()
        fbo_texture.release()
        return frames


def export(prog: stim.StimProgram, config):
    n_windows = len(config["windows"])
    win_outs = []
    for idx in range(n_windows):
        _logger.info(f"Exporting for window {idx + 1}/{n_windows}")
        # Use ArrayRenderer for offscreen GPU rendering
        renderer = ArrayRenderer(process_idx=1 + idx, config=config)
        frames, frame_times, triggers = renderer.render(prog)
        win_outs.append((frames, frame_times, triggers))

    # Get channel mapping, from array to windows.
    src_to_out = {}
    for w_idx, w in enumerate(config["windows"]):
        for c_idx, ch in enumerate(w["channels"]):
            src_to_out[ch] = [w_idx, c_idx]
    max_ch = max(src_to_out.keys())
    stim_shapes = [r[0].shape[0:3] for r in win_outs]
    assert all(
        s == stim_shapes[0] for s in stim_shapes
    ), "All windows must have the same (f, h, w) stimulus shape."
    f, h, w = stim_shapes[0]
    # Create the output stimulus array.
    frames = np.zeros((f, h, w, max_ch + 1), dtype=np.uint8)
    for ch in range(max_ch + 1):
        if ch in src_to_out:
            w_idx, c_idx = src_to_out[ch]
            frames[:, :, :, ch] = win_outs[w_idx][0][:, :, :, c_idx]
    # We need triggers and frame times from only the first window.
    frame_times = win_outs[0][1]
    frame_durs = np.diff(frame_times)
    assert len(frame_durs) == f, f"{len(frame_durs)=}, {f=}"
    triggers = win_outs[0][2]
    out_stim = fpspy.stim.StimArray(
        frames,
        frame_durations=frame_durs,
        zoom=1,
        triggers=triggers,
        label="exported_stimulus",
    )
    return out_stim


def validate_stim(stim_path, stim_config) -> bool:
    """Run checks on the stimulus before sending to presenter processes.

    Currently, it's just loading the program.

    Returns False on sucess, True on error.
    """
    try:
        prog = stim.create_program(stim_path, stim_config)
    except Exception as e:
        _logger.error(f"Failed to load stimulus program:\n{e}")
        #print(f"Failed to load stimulus program:\n{e}")
        return True
    return False


def start_presenter_processes(config, out_dir, delay, enable_triggers, log_level):
    """Start presenter processes for all windows and return them."""
    # Create queues for inter-process communication.
    n_windows = len(config["windows"])
    cmd_queues = [mp.Queue() for _ in range(n_windows)]
    status_queue = mp.Queue()
    processes = []
    for idx in range(1, len(cmd_queues) + 1):
        p = mp.Process(
            target=pyglet_app,
            args=(
                idx,
                config,
                out_dir,
                cmd_queues[idx - 1],
                status_queue,
                delay,
                # Only enable triggers for the first window.
                enable_triggers if idx == 1 else False,
                log_level,
            ),
        )
        p.start()
        processes.append(p)
        # Delay slightly to increase consistency of the window order in the OS.
        time.sleep(0.005) 
    return processes, cmd_queues, status_queue
