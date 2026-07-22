from queue import Queue
from typing import Any
import multiprocessing as mp
import fpspy.config
import fpspy.fps_queue
import fpspy._logging
import fpspy.arduino
import OpenGL.GL as gl
import fpspy.stim as stim
import fpspy.color
import dataclasses
from pathlib import Path
import numpy as np
import datetime
import csv
import logging
import time
from moderngl_window.conf import settings
import pyglet
from typing import Callable, Optional, Literal
import moderngl_window
from moderngl_window.context.base import WindowConfig, BaseWindow
from fpspy import errors as fpspy_errors
from fpspy.moderngl_helpers import probe_default_fbo_srgb
from fpspy.frame_handling import _wait_or_skip

# Globals
# Callback is given the frame index.
OnTriggerCallback = Callable[[], None]
_logger = logging.getLogger(__name__)

def present_live(
    process_idx: int,
    config: dict,
    out_dir: str | Any,
    cmd_queue: Queue,
    reply_queue: Queue,
    arduino_queue: Queue,
    delay:float,
    enable_triggers: bool,
    log_level="INFO",
):
    """
    Start a live runtime of a pyglet app

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
    reply_queue : multiprocessing.Queue
        This presenter's own queue for answering commands. Not shared with
        the other presenters, so the reader knows the sender from the queue.
    arduino_queue : multiprocessing.Queue
        Queue for the Arduino's asynchronous "done" event. Separate from
        reply_queue because it is not an answer to any command, and would
        otherwise desynchronise a reader counting command replies.
    delay : float
        Delay added before starting the stimulus presentation, in seconds.
    enable_triggers : bool
        Whether to enable Arduino triggers during the presentation.
    # log_level : str
    #     Logging level for this process (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    if not hasattr(pyglet, "canvas"):
        pyglet.canvas = pyglet.display

    presenter = Presenter(
        process_idx,
        config,
        out_dir,
        cmd_queue,
        reply_queue,
        delay=delay,
    )
    if enable_triggers:
        controller = fpspy.arduino.ArduinoController(
            port=fpspy.config.get_arduino_port(config),
            baud_rate=fpspy.config.get_arduino_baud_rate(config),
            trigger_command=fpspy.config.get_arduino_trigger_command(config),
            status_queue=arduino_queue,
            sender_idx=process_idx,
        )
        # Triggers fire synchronously on the render loop thread (right after
        # buffer swap). Play-start/stop and forwarded "device_cmd" commands
        # carry the slow-rate LED/trigger-mode semantics. The Presenter itself
        # stays Arduino-free.
        presenter.register_on_trigger(controller.on_trigger)
        presenter.register_on_play_start(controller.on_play_start)
        presenter.register_on_stop(controller.on_stop)
        presenter.register_device_handler(controller.handle_command)
    presenter.run_empty()



def start_presenter_processes(config, out_dir, delay, enable_triggers, log_level):
    """Start presenter processes for all windows and return them.

    Each presenter gets its own command queue and its own reply queue. The
    reply queues are deliberately not shared: a reply is always the answer to
    the command just sent, so one queue per presenter lets a caller wait for
    "one reply from each" without assuming an arrival order, and without
    replies from different windows being mistaken for one another.

    The Arduino queue is shared but has a single writer (only window 1 owns
    the Arduino), and carries only the asynchronous "arduino_done" event.
    """
    # Create queues for inter-process communication.
    n_windows = len(config["windows"])
    cmd_queues = [mp.Queue() for _ in range(n_windows)]
    reply_queues = [mp.Queue() for _ in range(n_windows)]
    arduino_queue = mp.Queue()
    processes = []
    for idx in range(1, len(cmd_queues) + 1):
        p = mp.Process(
            target=present_live,
            args=(
                idx,
                config,
                out_dir,
                cmd_queues[idx - 1],
                reply_queues[idx - 1],
                arduino_queue,
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
    return processes, cmd_queues, reply_queues, arduino_queue


# TODO:
@dataclasses.dataclass
class PlayState:
    """State needed to be stored when stepping through a stimulus via commands."""

    prog: stim.StimProgram
    frame_idxs: np.ndarray
    triggers: np.ndarray
    current_frame: int

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



class Presenter:
    """ Stimulus presenter

    This class is a wrapper around the pyglet window class and the moderngl_window
    BaseWindow class. It is responsible for loading and presenting the stimuli.
    The Presenter class is instantiated in a separate process for each window
    and communicates with the GUI process via a multiprocessing.Queue.
    """

    dropped_frames_filename_fmt = "dropped_frames_win{window_idx}.csv"
    frame_timings_filename_fmt = "frame_timings_win{window_idx}.npz"
    _window: BaseWindow | None = None
    _logger: logging.LoggerAdapter | None = None
    def __init__(
        self,
        process_idx: int,
        config: dict,
        out_dir: str,
        cmd_queue: Queue,
        reply_queue: Queue,
        delay: float=10.0,
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
        reply_queue : multiprocessing.Queue
            This presenter's own queue for answering commands on. One reply
            is emitted per command that has an answer.
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
        self.reply_queue = reply_queue
        self.out_dir = Path(out_dir)
        self.delay = delay
        self._logger: logging.LoggerAdapter | None = None
        self.setup_logging()
        self.c_channels = config["windows"][str(self.process_idx)]["channels"]
        self.mirror = config["windows"][str(self.process_idx)]["mirror"]
        self.rotation = config["windows"][str(self.process_idx)]["rotation"]
        self.fps = config["fps"]
        self.frame_duration = 1 / self.fps
        self.clear_rgba = config["windows"][str(self.process_idx)]["clear_rgba"]
        self.setup_window(config)
        # Callbacks. Currently only allows for one callback per event.
        # Purpose: to allow for arduino color changing.
        self._on_trigger: Callable | None = None
        self._on_stop: Callable |None = None
        self._on_play_start: Callable | None = None
        self._device_handler: Callable | None = None
        # Clear color currently in effect; "white_screen" swaps it to white
        # until the next stop restores the configured color.
        self.active_clear_rgba = self.clear_rgba
        # Play state is used when loading and stepping (not needed for one-shot play).
        self.play_state: Optional[PlayState] = None


    @property
    def window(self) -> BaseWindow:
        """
        moderngl Base window as used by the presenter
        """
        if self._window is None:
            raise fpspy_errors.NoBaseWindowError("no window; call create_window() first")
        return self._window

    @property
    def logger(self) -> logging.LoggerAdapter:
        """
        public logger property
        """
        if self._logger is None:
            raise RuntimeWarning("No logger connected")
        return self._logger

    def _is_step_play(self):
        """Check if we are in step-play mode."""
        return self.play_state is not None and self.play_state.current_frame >= 0

    def setup_window(self, config:dict):
        """
        Setup the window configured for moderngl. We write the required parameters to the settings config imported
        from moderngl.
        Parameters
         ----------
        config: dict
            The config dictionary created at startup of fpspy
        """

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
        if self.clear_rgba == [0.5, 0.5, 0.5, 1.0]:
            self.logger.warning(
                "Received {self.clear_rgba} as clear color. RGB values "
                "are interpreted as sRGB, so 0.5  will map to ~0.22 in "
                "intensity when displayed. 50% grey in sRGB is ~0.73."
            )
        self._window = moderngl_window.create_window_from_settings()

        self.window.position = (window_config["x_shift"], window_config["y_shift"])
        self.window.init_mgl_context()
        self.window.set_default_viewport()

    def setup_logging(self):
        """
        Sets up the logger which logs the presentation of stimuli
        """
        self._logger = logging.LoggerAdapter(
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

    def register_on_play_start(self, callback: Callable[[], None]):
        """Register a callback fired just before a play presentation starts."""
        self._on_play_start = callback

    def register_device_handler(self, callback: Callable):
        """Register a handler for "device_cmd" commands from the main process."""
        self._device_handler = callback

    def notify_stop(self):
        """Notify the stop event."""
        if self._on_stop is not None:
            self._on_stop()

    def notify_trigger(self):
        """Notify the trigger event."""
        if self._on_trigger is not None:
            self._on_trigger()

    def notify_play_start(self):
        """Notify the play-start event."""
        if self._on_play_start is not None:
            self._on_play_start()

    def run_empty(self):
        """Do nothing, waiting for commands from the main process."""
        self.window.use()
        while not self.window.is_closing:
            # Only clear/swap if no stimulus is loaded (otherwise show_frame handles it)
            if not self._is_step_play():
                self.window.ctx.clear(*self.active_clear_rgba)
                self.window.swap_buffers()
            else:
                # This next line is a proposed fix for the Windows first frame issue.
                self.window._window.dispatch_events()
            self.communicate()  # Check for commands from the main process (gui)
            time.sleep(0.001)  # Sleep for 1 ms to avoid busy waiting
        self.close_window()

    def reply(self, kind: str, **payload):
        """Answer the command currently being handled.

        Cheap enough for the render loop: Queue.put() hands off to the feeder
        thread and returns, so it cannot block on a slow reader.
        """
        fpspy.fps_queue.put_reply(self.reply_queue, kind, self.process_idx, **payload)

    def communicate(self):
        """Check and execute commands from the main process (gui)."""
        if self.queue.empty():
            return
        command = fpspy.fps_queue.get_from(self.queue)

        do_stop = False
        do_destroy = False
        match command.type:
            case "white_screen":
                # Display state only: the GUI orchestrates any accompanying
                # Arduino command separately (via "device_cmd").
                self.active_clear_rgba = (1.0, 1.0, 1.0, 1.0)
            case "device_cmd":
                if self._device_handler is not None:
                    self._device_handler(*command.args, **command.kwargs)
                else:
                    self.logger.warning(
                        f"device_cmd received but no handler registered: {command}"
                    )
            case "load":
                self.load(*command.args, **command.kwargs)
                # Send back total frame count for GUI (don't show frame yet)
                if self.play_state is not None:
                    total = len(self.play_state.frame_idxs)
                    self.reply("total_frames", total=total)
            case "step_next":
                if self.play_state is not None:
                    do_stop = self.step_next(**command.kwargs)
                    # current_frame is now the frame that is displayed
                    frame_shown = self.play_state.current_frame if self.play_state else -1
                    self.reply("stepped", frame=frame_shown)
                else:
                    self.reply("no_stimulus")
            case "step_prev":
                if self.play_state is not None:
                    do_stop = self.step_prev(**command.kwargs)
                    # current_frame is now the frame that is displayed (-1 for cleared)
                    frame_shown = self.play_state.current_frame
                    self.reply("stepped", frame=frame_shown)
                else:
                    self.reply("no_stimulus")
            case "play":
                do_stop = self.play(*command.args, **command.kwargs)
                # Answer regardless of whether the presenter is stopping: the
                # caller uses this to chain multiple stimuli in a sequence.
                self.reply("play_finished")
            case "stop":
                do_stop = True
            case "destroy":
                do_destroy = True
        if do_stop or do_destroy:
            self.active_clear_rgba = self.clear_rgba
            self.notify_stop()
            self.reply("done")
        if do_destroy:
            self.close_window()
        # shader_loop() polls this to know whether to abort mid-presentation.
        return do_stop or do_destroy

    def _load(self, stim_path, stim_config, loops, t0, speed):
        """Load a stimuli; shared by load() and play()."""
        prog = stim.create_program(stim_path, stim_config)
        self.logger.info(f"[start] program setup")
        s_frames, triggers = prog.setup(
            self.window.ctx, *self.window.size, self.c_channels,
            self.mirror, self.rotation, self.process_idx
        )
        self.logger.info(f"[end] program setup")
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
        stim_path : str or Path
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
        """Show the given frame.

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
        self.notify_play_start()
        dropped_frames = self.shader_loop(prog, frame_idxs, s_frames, triggers_arr)
        self.record_dropped_frames(dropped_frames)
        prog.cleanup()
        return close_after
    # TODO:
    def shader_loop(self, prog: stim.StimProgram, frame_idxs, s_frames, triggers):
        """
        Main loop for presenting the stimulus.
        """
        assert len(frame_idxs) == len(s_frames) - 1
        N = len(frame_idxs)
        self.window.use()
        srgb_capable, srgb_enabled = probe_default_fbo_srgb()
        self.logger.debug(
            f"Framebuffer sRGB capable: {srgb_capable}, enabled: {srgb_enabled}"
        )
        if srgb_enabled:
            raise RuntimeError(
                "Default framebuffer has sRGB enabled, which means writing to it will "
                "apply a linear->sRGB conversion. This is not compatible with fpspy "
                "which promises to send program outputs to the framebuffer as-is."
            )
        dropped_frames = []
        # Per-frame timestamps: loop-start, after-wait, after-render, after-swap.
        # Preallocated and written by index only — no allocation, formatting or
        # I/O inside the loop, so measuring doesn't perturb the frame schedule.
        # Skipped (and never-reached) frames stay NaN. All stats and file
        # writing happen in _save_timings(), after the loop.
        timings = np.full((N, 4), np.nan, dtype=np.float64)
        for i in range(N):
            timings[i, 0] = time.perf_counter()
            is_exit = self.communicate()
            if is_exit:
                self._save_timings(timings, s_frames)
                return dropped_frames

            # Sync frame presentation to the scheduled time.
            next_frame_time = s_frames[i + 1] if i < N - 1 else None
            skip = _wait_or_skip(s_frames[i], next_frame_time, self.fps, self.logger)
            if skip:
                dropped_frames.append(i)
                continue
            timings[i, 1] = time.perf_counter()

            # Clear window (to black is fine), render the stimulus and swap buffers.
            self.window.ctx.clear(0, 0, 0)
            prog.render(self.window.ctx, frame_idxs[i], i)
            timings[i, 2] = time.perf_counter()
            self.window.swap_buffers()
            timings[i, 3] = time.perf_counter()
            if triggers[i]:
                self.notify_trigger()
        self._save_timings(timings, s_frames)
        return dropped_frames

    def _save_timings(self, timings, s_frames):
        """Save per-frame timestamps to disk and log a one-shot summary.

        Columns of `timings` (time.perf_counter() seconds): loop-start,
        after-wait, after-render, after-swap. NaN rows are frames that were
        skipped or never reached. `s_frames` (len N+1) is saved alongside so
        lateness and remaining budget can be recomputed offline.
        """
        out_path = self.out_dir / self.frame_timings_filename_fmt.format(
            window_idx=self.process_idx
        )
        np.savez(out_path, timings=timings, s_frames=s_frames)

        shown = ~np.isnan(timings[:, 3])
        n_shown = np.count_nonzero(shown)
        if n_shown == 0:
            self.logger.warning(f"No frames shown. Timings saved to {out_path}")
            return
        t = timings[shown]
        render_time = t[:, 2] - t[:, 1]
        swap_wait = t[:, 3] - t[:, 2]
        lateness = t[:, 1] - s_frames[:-1][shown]
        budget_left = s_frames[1:][shown] - t[:, 3]

        def fmt(x):
            return (
                f"median {np.median(x) * 1e3:.2f} ms, "
                f"p99 {np.percentile(x, 99) * 1e3:.2f} ms, "
                f"max {np.max(x) * 1e3:.2f} ms"
            )

        self.logger.info(
            f"Frame timings ({n_shown}/{len(timings)} frames shown, "
            f"saved to {out_path}):\n"
            f"\trender:      {fmt(render_time)}\n"
            f"\tswap wait:   {fmt(swap_wait)}\n"
            f"\tlateness:    {fmt(lateness)}\n"
            f"\tbudget left: min {np.min(budget_left) * 1e3:.2f} ms, "
            f"median {np.median(budget_left) * 1e3:.2f} ms"
        )

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