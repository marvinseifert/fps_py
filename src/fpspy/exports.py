from fpspy import stim
import numpy as np
import moderngl
import fpspy.color
import logging

# Setup logger, propagates to the root logger configured in _logging.setup_main_logging()"
# So essentially, this logger is main process always by default.
_logger = logging.getLogger(__name__)

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
            self.ctx,
            *self.window_size,
            self.c_channels,
            self.mirror,
            self.rotation,
            self.process_idx,
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


def export(prog: stim.StimProgram, config, _logger):
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
    out_stim = stim.StimArray(
        frames,
        frame_durations=frame_durs,
        zoom=1,
        triggers=triggers,
        label="exported_stimulus",
    )
    return out_stim