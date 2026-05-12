import math
import logging
import time
from pathlib import Path
from typing import Protocol, Optional, Sequence, Tuple
import importlib.resources
import json
import h5py
import hdf5plugin
import numpy as np
import einops
import moderngl
import importlib
import importlib.util
import copy
import numpy.typing as npt
import warnings
from typing import Any, Dict


_logger = logging.getLogger(__name__)

__all__ = [
    "StimArray",
    "TextureSequence",
    "ProceduralShader",
    "MoviePlayer",
    "StimProgram",
    "CURRENT_HDF5_FORMAT_VER",
]

MOVIE_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm")


CURRENT_HDF5_FORMAT_VER = "1"
TEXTURE_FRAG_SHADER = "fragment_shader_colour.glsl"
QUAD_VERTEX_SHADER = "vertex_shader.glsl"


def loop_triggers(triggers, n_frames, n_loops):
    trigger_repeats = [triggers]
    for i in range(1, n_loops):
        trigger_repeats.append(triggers + i * n_frames)
    triggers_out = np.concatenate(trigger_repeats)
    triggers_out = triggers_out.astype(int)
    return triggers_out


def loop(s_frames, triggers, n_loops):
    """Repeat the frames and triggers.

    Parameters
    ----------
    s_frames : np.ndarray
        Start times of each frame and the end time of the last frame, in seconds.
    triggers : np.ndarray
        Frame indices where triggers occur.

    Returns
    -------
    frame_idxs : np.ndarray
        Frame indices.
    s_frames : np.ndarray
        Frame schedule.
    s_triggers : np.ndarray
        Frame indices where triggers occur.
    """
    if triggers is None or triggers.size == 0:
        raise ValueError("Triggers must be provided.")
    n_frames = len(s_frames) - 1
    if len(triggers) > n_frames:
        raise ValueError("More triggers than frames, {len(triggers)=} > {n_frames=}")
    period = s_frames[-1] - s_frames[0]
    start_times = s_frames[:-1]

    # Frame indices.
    idxs_out = np.tile(np.arange(len(start_times)), n_loops)
    # Trigger indices.
    triggers_out = loop_triggers(triggers, n_frames, n_loops)
    # Frame schedule.
    s_frame_repeats = [start_times]
    for i in range(1, n_loops):
        s_frame_repeats.append(start_times + i * period)
    last_frame = period * n_loops + s_frames[0]
    s_frames_out = np.concatenate(s_frame_repeats + [np.array([last_frame])])
    assert len(idxs_out) + 1 == len(s_frames_out)
    assert len(triggers_out) == len(triggers) * n_loops
    return idxs_out, s_frames_out, triggers_out


def decompress_triggers(triggers, n_frames):
    """Decompress trigger indices into a boolean array.

    Parameters
    ----------
    triggers : np.ndarray
        Frame indices where triggers occur.
    n_frames : int
        Total number of frames.

    Returns
    -------
    np.ndarray
        Boolean array indicating trigger frames.
    """
    trigger_array = np.zeros(n_frames, dtype=bool)
    trigger_array[triggers] = True
    return trigger_array


def delay(s_frames, delay):
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


def create_centered_quad(
    stim_width, stim_height, win_width, win_height, mirror: bool, rotation: float
):
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
    mirror : bool
        Whether to mirror the stimulus horizontally.
    rotation : float
        The rotation (degrees) to apply to the stimulus (after any mirror).

    Returns
    -------
    tuple
        A tuple of scaling factors (scale_x, scale_y) and the quad vertices array.
    """
    # Calculate the aspect ratio of the window and the texture
    _logger.debug(
        f"stim shape (w, h): ({stim_width}, {stim_height}), "
        f"window shape (w, h): ({win_width}, {win_height}), "
    )
    # If the stimulus has shape (f, h, w, c) == (f, 1, 1, c), we broadcast by
    # having the single pixel cover the whole screen.
    do_broadcast = stim_height == 1 and stim_width == 1
    if do_broadcast:
        # Fullscreen with no sense of orientation.
        # fmt: off
        quad = np.array(
            [  # x   y
               [-1,  1],  # top left
               [-1, -1],  # bottom left
               [ 1,  1],  # top right
               [ 1,  1],  # top right
               [-1, -1],  # bottom left
               [ 1, -1],  # bottom right
            ],
            dtype=np.float32,
        )
        # fmt: on
    else:
        hh = stim_height / 2.0
        hw = stim_width / 2.0
        # fmt: off
        quad = np.array(
            [  #  x    y
               [-hw,  hh],  # top left
               [-hw, -hh],  # bottom left
               [ hw,  hh],  # top right
               [ hw,  hh],  # top right
               [-hw, -hh],  # bottom left
               [ hw, -hh],  # bottom right
            ],
            dtype=np.float32,
        )
        # fmt: on
        # We must consider window aspect ratio, and stimulus mirror and rotation.
        # Determine scaling factors based on aspect ratios
        # Scale from stimulus pixels to NDC space (-1 to 1)
        x_scale = 2 / win_width
        y_scale = 2 / win_height
        mirror_transform = np.array(
            [
                [-1 if mirror else 1, 0],
                [0, 1],
            ]
        )
        rotate_transform = np.array(
            [
                [np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))],
                [np.sin(np.radians(rotation)), np.cos(np.radians(rotation))],
            ]
        )
        # Scale down from full window to stimulus size.
        scale = np.array(
            [
                [x_scale, 0],
                [0, y_scale],
            ]
        )
        transform = np.eye(2)
        if mirror:
            transform = mirror_transform @ transform
        if rotation != 0:
            transform = rotate_transform @ transform
        transform = scale @ transform
        if np.any(transform > 1.0):
            _logger.info(
                f"Stimulus is larger than window and will be clipped. {transform=}"
            )
        _logger.debug(f"{transform=}")
        # Apply
        quad = (transform @ quad.T).T
    # Insure the quad is centered at the origin.
    xmin = np.min(quad[:, 0])
    xmax = np.max(quad[:, 0])
    ymin = np.min(quad[:, 1])
    ymax = np.max(quad[:, 1])
    in_centered = math.isclose(xmin + xmax, 0.0, abs_tol=1e-6) and math.isclose(
        ymin + ymax, 0.0, abs_tol=1e-6
    )
    assert in_centered, f"Quad is not centered at origin. {quad=}"

    quad = einops.rearrange(quad, "v c -> (v c)")
    # Array must be contiguous float32, in anticipation of calling asbytes().
    quad = np.ascontiguousarray(quad, dtype=np.float32)
    _logger.debug(f"{quad=}")
    return quad


def _set_uniforms(program, uniforms: dict):
    """Set shader uniforms from a dictionary.

    Handles conversion of lists to tuples for vec uniforms.

    Parameters
    ----------
    program : moderngl.Program
        The shader program.
    uniforms : dict
        Dictionary mapping uniform names to values.
    """
    for name, value in uniforms.items():
        # Convert lists to tuples (needed for vec2, vec3, etc.).
        if isinstance(value, list):
            value = tuple(value)
        program[name].value = value


def spf_to_frame_times(spf, n_frames):
    """Convert seconds per frame to frame times array."""
    frame_times = np.arange(n_frames + 1) * spf
    return frame_times


def _expand_frame_durations(durs):
    """View of frame durations with shape broadcastable with (N, F, H, W, C).

    Possible outputs:
        - a scalar will produce (1, 1, 1, 1, 1)
        - (F,) will produce (1, F, 1, 1, 1)
        - (N, F) will produce (N, F, 1, 1, 1)

    The leading 1s are not strictly needed but make things easier to reason about.

    Static/class method so that it can be used to preview frame times from HDF5
    files without needing to deserialize the full stimulus object.
    """
    if np.ndim(durs) == 0:
        durs = np.array([durs])
    durs = durs.reshape(durs.shape + (1, 1, 1))  # add (..H, W, C) dims.
    if durs.ndim == 4:
        durs = einops.rearrange(durs, "f h w c -> 1 f h w c", h=1, w=1, c=1)
    return durs


def _frame_times(frame_durations, N, F):
    """Get frame times, in seconds as a 1D array of shape (N*F,).

    Returned array is float64, as float32 can hit precision issues for not
    unreasonably long stimuli.

    Static/class method so that it can be used to preview frame times from HDF5
    files without needing to deserialize the full stimulus object.
    """
    durs = _expand_frame_durations(frame_durations)
    durs = np.broadcast_to(durs, (N, F, 1, 1, 1))
    durs = durs.flatten()
    frame_times = np.concatenate([(0,), np.cumsum(durs, dtype=np.float64)])
    assert len(frame_times) == N * F + 1, f"{len(frame_times)=}, expected {N*F+1}"
    return frame_times


class StimArray:
    """The data of an array-based stimulus.

    StimArray can be considered the datum of a TextureSequence stimulus program. The
    main purpose of this class is to encapsulate serialization and deserialization.

    Goals:
        - act as the datum of a TextureSequence stimulus program.
        - save/load from hdf5 files, with support for versioning and backward
        compatibility.
        - reduce the disk and memory footprint of stimuli

    ## Memory and disk space
    Broadcasting and zooming are used to inflate stimuli, allowing them to be encoded
    in smaller arrays.

    ### Broadcasting and channel mask
    The nature of many stimuli, such as being full-field or monochrome, means that
    broadcasting can be utilized to significantly reduce the number of array elements
    needed to represent a stimuli.

    There are two ways in which broadcasting takes effect:

        1. Spatial and channel expansion. The H, W and C dimensions of the frames array
            are broadcast to the H, W and C dimensions of a display window. This means
            that full-field stimuli can be stored as (F, 1, 1, C) arrays, and monochrome
            stimuli can be stored as (F, H, W, 1) arrays.
        2. To be able to specify which LEDs to use with a monochrome stimuli, we use
            the `channel_mask` parameter. For example a [0, 1, 1, 0, 0] mask would cause
            a (F, H, W, 1) stimulus to be broadcast to (F, H, W, 6), where only the 2nd
            and 3rd channels are ever ON. There is more. The channel mask has up to 3
            dimensions, (N, F, C). The F dimension allows a channel mask to be specified
            per frame, or broadcast across all frames if F=1. The N dimension allows for
            multiple repeats of the frame array, each repeat using a different channel
            mask. In theory, the (F, H, W, C) frames array is broadcast to (N, F, H, W,
            C). The resulting stimulus can be thought of as

                frames          x   channel_mask
                (1, F, H, W, C) x (N, F, 1, 1, C) → (N, F, H, W, C)

    The broadcasting allows simple numpy operations to reinflate a stimulus from the two
    arrays—the frames and the channel mask. To keep this simplicity, there is no
    support for having any separate gap between the repeats of the frames. If you wish
    for there to be gaps, then this should be encoded in the frames array itself.

    If you need more complexity, it might be worth considering making a new StimProgram
    class and a new data class.

    Note that the channels masked out by a channel_mask are off (0) and not say 50%
    intensity. A separate parameter would be needed to support that.

    ### Zoom
    A stimulus is zoomed on presentation, based on the `zoom` parameter. An example
    use case is having a 80x50 single pixel noise stimulus be zoomed by 16x to fill
    a 1280x800 display with a 16x16 grid of squares.

    ## Conventions
    An array of shape (F, H, W, C) corresponds to:
        - F frames
        - (H, W) spatial dimensions, with (0, 0) corresponding to the top-left corner.
        - C channels
        - uint8 values in [0, 255]. The values are **not linear**, but interpreted as
          sRGB values. The standard conversion from linear to sRGB is:

          ```python
          np.where(arr <= 0.0031308, arr * 12.92, 1.055 * (arr ** (1 / 2.4)) - 0.055)
           ```

          Nearly all displays (including the light crafters) expect these sRGB values
          as input and will automatically decode them, so it is a mistake to store
          stimuli with a linear scale. Storing linear values with more bits, such as
          float16 or float32, and then converting to sRGB before sending to the display
          is possible; however, this will increase the disk footprint and is not
          currently supported by StimArray.

    It is important to note that OpenGL textures have (0, 0) correspond to the
    bottom-left. The TextureSequence stimulus program will vertically flip frames in
    order to maintaing the convention that (0, 0) is the top-left corner.
    """

    _frames: np.ndarray
    _frame_durations: np.ndarray | float
    _triggers: np.ndarray | None
    _channel_mask: np.ndarray | None
    _label: str | None
    metadata: dict

    def __init__(
        self,
        frames: np.ndarray,
        frame_durations: npt.ArrayLike,
        zoom: int,
        triggers: npt.ArrayLike | None = None,
        channel_mask: npt.ArrayLike | None = None,
        label: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        frames : np.ndarray
            4D numpy array with shape (num_frames, height, width, channels). If the
            channel dimension is 1, the stimulus will be broadcast to all channels of
            the display(s).
        frame_durations : array_like
            A single float will represent a shared frame duration (seconds). A
            (F,) array will specifies each frame duration. If there are any repeats
            implied by the channel mask, (F,) is broadcasted to (N, F). Finally, you can
            specify a full (N, F) array to specify the frame durations across all mask
            repeats. Frame durations rather than frame times are used as the latter has
            numerical precision issues. For example, if using float32, then the unit
            in the last place by t=16384s (~4 1/2 hours) is 1/512, so that the minimum
            distinguishable time difference is about 2ms. So for long stimuli, frame
            times would become noticeably inaccurate.
        zoom: int >= 1
            Zoom factor for displaying the stimulus.
        triggers : array_like, optional
            A 1D list/array with (F,) effective shape. The elements should be integers
            corresponding to frame indices. If None, trigger on every frame. Triggers
            will be the same for any repeat implied by the channel_mask.
        channel_mask : np.ndarray, optional
            An optional  boolean array of shape (C,) or (F, C) or (N, F, C) indicating
            which channels to use. Masked out channels (mask=0) will be set to zero when
            displayed. The F dimension corresponds to frames, and if F=1, it will be
            broadcast across all frames. The N dimension corresponds to repeats of the
            whole frame array, and allows for different channel masks to be applied in
            succession. If channel_mask is None, no mask is applied. For broadcasting
            purposes, (N, F, C) is effectively (N, F, 1, 1, C), such that the mask is
            applied uniformly across all HxW pixels.
        label : str, optional
            An optional label for the stimulus.
        metadata : dict, optional
            An optional dictionary of metadata. Things like checkerboard size
            can go in here, which may not be applicable to all stimuli.
            The metadata should be a non-nested dictionary with key-value pairs
            that are serializable with hdf5.
        """
        # _frames is np.ndarray (in-memory construction) or h5py.Dataset (lazy from
        # HDF5). Both support .shape/.dtype/.ndim/__len__/__getitem__, which is all
        # the hot path uses. broadcasted_view() materializes when numpy ops are
        # needed. When _frames is a Dataset, it holds a reference to its parent
        # h5py.File, which keeps the file open until self is GC'd or .close()'d.
        self._frames = frames
        self._frame_durations = np.asarray(frame_durations)
        # Reduce to scalar if possible, so that there is more certainty about the type.
        if np.size(self._frame_durations) == 1:
            self._frame_durations = self._frame_durations.item()
        self.zoom = int(zoom)
        self._triggers = np.asarray(triggers) if triggers is not None else None
        self._channel_mask = (
            np.asarray(channel_mask) if channel_mask is not None else None
        )
        self.label = label
        self.metadata = metadata if metadata is not None else {}
        self._validate_state()

    def _validate_state(self):
        """Raise a ValueError if there object state is found to be invalid."""
        if self._frames.ndim != 4:
            raise ValueError(f"Expected shape (f, h, w, c). {self._frames.shape=}.")
        if self.zoom < 1:
            raise ValueError(f"Zoom must be >= 1. {self.zoom=}.")
        if int(self.zoom) != self.zoom:
            raise ValueError(f"Zoom must be integer. {self.zoom=}.")
        if self._triggers is not None:
            if self._triggers.ndim != 1:
                raise ValueError(f"Expected 1D triggers. {self._triggers.shape=}.")
            if len(self._triggers) == 0:
                # Defensive. Having empty arrays is error prone as testing triggers!=None
                # doesn't guarantee we have triggers.
                raise ValueError("Triggers cannot be empty array. Use None instead.")
            is_sorted = np.all(self._triggers[:-1] <= self._triggers[1:])
            if not is_sorted:
                raise ValueError(f"Triggers must be sorted. {self._triggers=}.")
            if self._triggers[0] < 0:
                raise ValueError(f"Triggers must be non-negative. {self._triggers=}.")
            if self._triggers[-1] >= len(self._frames):
                raise ValueError(
                    f"Triggers must be less than number of frames. {self._triggers=}, "
                    f"{len(self._frames)=}."
                )
        if self._channel_mask is not None:
            if self._channel_mask.ndim > 3 or self._channel_mask.ndim == 0:
                raise ValueError(
                    f"Expected channel_mask with 1, 2 or 3 dims. "
                    f"{self._channel_mask.shape=}"
                )
            elif self._channel_mask.ndim in (2, 3):
                if self._channel_mask.shape[-2] not in (1, len(self._frames)):
                    raise ValueError(
                        f"Expected channel_mask with F={len(self._frames)} or F=1. Got "
                        f"{self._channel_mask.shape=}"
                    )
        if not np.ndim(self._frame_durations) in (0, 1, 2):
            raise ValueError(
                f"Expected frame_durations to be a 0, 1 or 2D. "
                f"{self._frame_durations=}, {np.shape(self._frame_durations)=}"
            )

        durs = _expand_frame_durations(self._frame_durations)
        try:
            np.broadcast_shapes(durs.shape, self.shape)
        except ValueError as e:
            raise ValueError(
                f"Frame durations ({np.shape(self._frame_durations)}) is not "
                f" broadcastable to stimulus shape ({self.shape})."
            ) from e

    def _frame_durations_is_scalar(self):
        """Return True if frame_times is a single float (fps)."""
        res = np.ndim(self._frame_durations) == 0
        return res

    @property
    def shape(self):
        res = np.broadcast_shapes(self._frames.shape, self.channel_mask().shape)
        assert len(res) == 5, f"Expected (N, F, H, W, C). Got {res=}"
        return res

    def broadcasted_view(self):
        """Get frames as shape (N, F, H, W, C) and channel mask as shape (N, F, 1, 1, C).

        Materializes lazy (h5py.Dataset-backed) frames to a full numpy array;
        cheap no-op if _frames is already ndarray. Callers that only need one
        frame should use frame_at() instead.
        """
        if isinstance(self._frames, h5py.Dataset):
            _logger.info(
                f"Materializing all frames from h5py.Dataset to ndarray "
                f"(shape={self._frames.shape}, dtype={self._frames.dtype})."
            )
        frames = np.asarray(self._frames)
        assert frames.ndim == 4
        F1, H, W, C1 = frames.shape
        channel_mask = self.channel_mask()
        N, F2, _, _, C2 = channel_mask.shape
        # The frames dimension decides F, and the channel mask decides C.
        assert F2 == 1 or F2 == F1, f"Frame dim is incompatible. {F1=}, {F2=}"
        assert C1 == 1 or C1 == C2, f"Channel dim is incompatible. {C1=}, {C2=}"
        C = C2
        F = F1
        frames = np.broadcast_to(frames, (N, F, H, W, C))
        channel_mask = np.broadcast_to(channel_mask, (N, F, 1, 1, C))
        return frames, channel_mask

    def frame_at(self, frame_idx):
        """Index into the stimulus as an (N x F, H, W, C) array.

        Lazy-friendly: reads exactly one frame from _frames (one chunk if backed
        by h5py.Dataset) and applies the channel mask. Does not materialize the
        full frame array.
        """
        N, F, _, _, C = self.shape
        f = frame_idx % F  # frame index within the (possibly repeated) frames array.
        # This next line is where we read to create a numpy array.
        frame = self._frames[f]
        assert isinstance(frame, np.ndarray), f"Should be a np.ndarray. {type(frame)=}"
        mask = self.channel_mask()
        mask = np.broadcast_to(mask, (N, F, 1, 1, C))
        per_frame_mask = einops.rearrange(mask, "n f 1 1 c -> (n f) 1 1 c")[frame_idx]
        # A more direct approach would be:
        #   m_f = 0 if mask.shape[1] == 1 else f
        #   per_frame_mask = mask[n, m_f, 0, 0]
        return frame * per_frame_mask

    # Previously, frame_at() was tidier, but it materialized a full frame numpy array.
    # def _frame_at_old(self, frame_idx):
    #     frames, channel_mask = self.broadcasted_view()
    #     frame = einops.rearrange(frames, "n f h w c -> (n f) h w c")[frame_idx]
    #     mask = einops.rearrange(channel_mask, "n f 1 1 c -> (n f) 1 1 c")[frame_idx]
    #     masked_frame = frame * mask
    #     return masked_frame

    def masked_frames(self):
        """Apply channel mask to frames, resulting in an (N, F, H, W, C) array.

        This will create a new array, possibly a very large one. Use broadcasted_view()
        if you would prefer to work with the frames and channel_mask as views of
        shape (N, F, H, W, C) and (N, F, 1, 1, C) respectively, which, under the hood,
        are much smaller arrays, possibly as small as (F, H, W, 1) and (C,).
        """
        frames, channel_mask = self.broadcasted_view()
        _logger.info(
            f"[start] Expanding frames with channel mask "
            f"{frames.shape} x {channel_mask.shape} -> {self.shape}"
        )
        masked_frames = frames * channel_mask
        _logger.info(f"[end] Expanding frames with channel mask")
        assert masked_frames.ndim == 5, f"{masked_frames.shape=}"
        return masked_frames

    @property
    def height(self):
        return self._frames.shape[1]

    @property
    def width(self):
        return self._frames.shape[2]

    @property
    def n_channels(self):
        """The number of channels, after any expandion due to channel mask."""
        n_ch = self.shape[-1]
        return n_ch

    def channel_mask(self) -> np.ndarray:
        """The effective channel mask after broadcasting.

        We need to insert (1,1) for the (H, W) dimensions. As they are in the middle,
        we can't rely on broadcasting to implicitly map the dimensions. If we stored
        everything as channel first, then we could just rely on broadcasting.
        """
        if self._channel_mask is None:
            frames_nch = self._frames.shape[-1]
            mask = np.ones((1, 1, 1, 1, frames_nch), dtype=bool)
        elif self._channel_mask.ndim == 1:
            mask = einops.rearrange(self._channel_mask, "c -> 1 1 1 1 c")
        elif self._channel_mask.ndim == 2:
            mask = einops.rearrange(self._channel_mask, "f c -> 1 f 1 1 c")
        elif self._channel_mask.ndim == 3:
            mask = einops.rearrange(self._channel_mask, "n f c -> n f 1 1 c")
        else:
            assert False, f"{self._channel_mask.shape=}"
        return mask

    def n_mask_repeats(self):
        """The number of repeats of the frames array implied by the channel mask."""
        N, F, H, W, C = self.channel_mask().shape
        return N

    def total_frames(self):
        """The total number of frames, accounting for channel mask repeats."""
        if self._channel_mask is None:
            n_frames = len(self._frames)
        else:
            n_frames = self.n_mask_repeats() * len(self._frames)
        return n_frames

    def fps(self, allow_estimate: bool):
        """Return the stimulus fps (possibly estimated), or None if not fixed.

        Frames have fixed fps if frame_durations is a scalar. Frames _might_ have a
        fixed fps if frame_durations is an array with equal elements.

        None is returned if there isn't a fixed fps (non-scalar or allow_estimate=True
        but unequal frame durations).
        """
        res = None
        if self._frame_durations_is_scalar():
            res = 1.0 / self._frame_durations
        else:
            if allow_estimate:
                # Are all frame times evenly spaced?
                dur0 = self._frame_durations[0]
                atol = 1e-4  # 100 microseconds.
                if np.allclose(self._frame_durations, dur0, rtol=0, atol=atol):
                    res = 1.0 / dur0
        return res

    def frame_times(self) -> np.ndarray:
        """Get frame times, in seconds as a 1D array of shape (N*F,).

        Returned array is float64, as float32 can hit precision issues for not
        unreasonably long stimuli.
        """
        N, F, _, _, _ = self.shape
        frame_times = _frame_times(self._frame_durations, N, F)
        return frame_times

    def frame_start_times(self) -> np.ndarray:
        start_and_final = self.frame_times()
        return start_and_final[:-1]

    def triggers(self):
        """Get triggers as a 1D array of shape (N*F,)."""
        N, F, _, _, _ = self.shape
        if self._triggers is None:
            triggers = np.arange(len(self._frames), dtype=np.uint64)
        else:
            triggers = self._triggers
        assert triggers.ndim == 1, f"Only supports (F,) arrays {triggers.shape=}"
        triggers = loop_triggers(triggers, F, N)
        return triggers

    def __repr__(self):
        tr_str = (
            "None" if self._triggers is None else f"len(triggers)={len(self._triggers)}"
        )
        return (
            f"StimArray(shape(frames)={self._frames.shape}, "
            f"fps={self.fps(allow_estimate=True):.3f}, {tr_str}, "
            f"zoom={self.zoom}, label={self.label}, metadata={self.metadata.__repr__()}"
        )

    def with_channels(self, channels):
        """Return a new Stim object with only the specified channels.

        Parameters
        ----------
        channels : list of int
            The channels to keep.

        Returns
        -------
        Stim
            A new Stim object with only the specified channels.
        """
        assert self._frames.ndim == 4, f"{self._frames.shape=}"
        F, H, W, C = self._frames.shape
        has_mask = self._channel_mask is not None
        new_channel_mask = self._channel_mask[..., channels] if has_mask else None
        # If frames has shape (F, H, W, 1) (i.e. c=1), then it is monochrome and the
        # channel mask will inflate this and determine which channels are available.
        if C == 1:
            requesting_only_ch0 = set(channels) == {0}
            if not has_mask and not requesting_only_ch0:
                raise ValueError(
                    f"Frames has shape {self._frames.shape}, but requesting channels "
                    "{channels}."
                )
            # Frames is used as-is.
            new_frames = self._frames
        else:
            # This covers two cases:
            # 1. channel_mask is None (what we anticipate as being most common)
            # 2. There is also a channel_mask. This is supported, but may be niche.
            new_frames = self._frames[:, :, :, channels]
        return StimArray(
            frames=new_frames,
            frame_durations=self._frame_durations,
            zoom=self.zoom,
            triggers=self._triggers,
            channel_mask=new_channel_mask,
            label=self.label,
            metadata=self.metadata,
        )

    def write_hdf5(self, path: Path, dataset_opts: Optional[dict] = None):
        """Write the stimulus to an HDF5 file.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file where the stimulus metadata will be stored.
        """
        with h5py.File(path, "w") as f:
            _write_hdf5_v1(self, f, dataset_opts)

    def close(self):
        """Close the underlying HDF5 file if this StimArray is HDF5-backed.

        Idempotent. For in-memory StimArrays this is a no-op.
        """
        if isinstance(self._frames, h5py.Dataset):
            self._frames.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def read_hdf5(path: Path):
        """Read a stimulus from an HDF5 file.

        For format v1, the returned StimArray is lazy: `_frames` is an
        `h5py.Dataset` and the HDF5 file remains open. The Dataset keeps the
        File alive via Python refs; call `.close()` (or use as a context
        manager) to release the file promptly.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file from which to read the stimulus metadata.

        Returns
        -------
        StimArray
            The stimulus object created from the HDF5 file.
        """
        f = h5py.File(path, "r")#, rdcc_nbytes=10 * 1024 * 1024)  # 10MB chunk cache.
        try:
            ver = f.attrs.get("format_version", "0")
            if ver == "0":
                # v0 reads eagerly; file is no longer needed after _read_hdf5_v0
                # returns. Close it here.
                stim = _read_hdf5_v0(f)
                f.close()
                return stim
            elif ver == "1":
                # File stays open. Dataset in the returned StimArray keeps it
                # alive until close()/GC.
                return _read_hdf5_v1(f)
            else:
                raise ValueError(f"Unsupported HDF5 format version: {ver}")
        except BaseException:
            f.close()
            raise

    @staticmethod
    def preview_hdf5(path: Path):
        info = {}
        with h5py.File(path, "r") as f:
            ver = f.attrs.get("format_version", "0")
            if ver == "0":
                info["version"] = 0
                info.update(_preview_hdf5_v0(f))
            elif ver == "1":
                info["version"] = 1
                info.update(_preview_hdf5_v1(f))
            else:
                raise ValueError(f"Unsupported HDF5 format version: {ver}")
        return info

    @staticmethod
    def frame_times_from_hdf5(path: Path) -> np.ndarray:
        """Preview frame times from HDF5 file without loading full stimulus."""
        with h5py.File(path, "r") as f:
            ver = f.attrs.get("format_version", "0")
            if ver == "0":
                frame_times = frame_times_from_hdf5_v0(f)
            elif ver == "1":
                frame_times = frame_times_from_hdf5_v1(f)
            else:
                raise ValueError(f"Unsupported HDF5 format version: {ver}")
        assert isinstance(frame_times, np.ndarray)
        return frame_times


def _chunk_shape(arr, max_chunk_bytes=1 * 1024 * 1024):
    """Calculate an array's chunk shape so that each chunk fills up to max_chunk_bytes.

    Not using a good chunk shape can lead to very slow read performance. For example,
    if the chuck is too big, then each read will involve reading and decompressing a
    large amount of data, and if it doesn't fit in the hdf5 chunk cache, most of the
    chunk will be evicted before the next read, possibly leading to the same data
    being loaded and decompressed multiple times.

    If you increase the chunk size being saved, you should also increase the chunk cache
    size when reading. The default chunk cache size is 1MB.

    If a single frame doesn't fit in the max_chunk_bytes, there is a warning, but no
    error; instead, a best effort is made, and the chunk is set to the size of a single
    frame.
    """
    frame_bytes = max(1, int(np.prod(arr.shape[1:]) * arr.dtype.itemsize))
    if frame_bytes > max_chunk_bytes:
        _logger.warning(
            f"Single frame is larger than max chunk size. {frame_bytes=}, "
            f"{max_chunk_bytes=}. Consider increasing max_chunk_bytes. "
            f"{frame_bytes / (2**20):.2f} MiB per frame. This will be used for the "
            f"chunk size. For good read performance, ensure the chunk cache is larger "
            f" than this."
        )
        n = 1
    else:
        n = min(arr.shape[0], int(max_chunk_bytes // frame_bytes))
    assert n > 0
    return (n, *arr.shape[1:])


def _write_hdf5_v1(stim: StimArray, f, dataset_opts=None):
    if dataset_opts is None:
        dataset_opts = {
            # zstd at clevel=9 gives ~2700x compression on highly-redundant frame data
            # (e.g. moving bars on black: 3 GB -> 1.15 MB), with per-frame decode
            # under 12 ms and no 20+ ms outliers. blosclz/lz4 decode faster on
            # already-cached data but have occasional 20+ ms tail latencies and
            # 12x larger files on this kind of stim.
            **hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE),

            # Alternatives, kept for reference:
            # **hdf5plugin.Blosc(cname="blosclz", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
            #   ~2x larger files than zstd cl=9, slightly faster median (9.65 vs 9.81 ms)
            #   but worse worst-case (max ~21 ms vs ~11 ms).
            # **hdf5plugin.Blosc(cname="lz4hc",   clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
            #   ~12x larger than zstd cl=9, similar worst-case (~11 ms).
        }

    def filter_opts(arr, dataset_opts):
        """Prevent errors on empty arrays."""
        res = copy.deepcopy(dataset_opts)
        # The last check covers isinstance(arr, h5py.Empty):
        if np.isscalar(arr) or arr.size == 0 or arr.shape is None:
            res.pop("compression", None)
            res.pop("chunks", None)
            res.pop("shuffle", None)
            res.pop("compression_opts", None)
        return res

    def chunk_opts(arr):
        shape = getattr(arr, "shape", None)
        if not shape:
            return {}
        else:
            assert 0 not in shape, f"Got zero size dimension. {shape=}"
            return {"chunks": _chunk_shape(arr)}

    f.attrs["format_version"] = "1"
    label = stim.label if stim.label is not None else h5py.Empty("f")
    if stim._triggers is None or len(stim._triggers) == 0:
        triggers = h5py.Empty("f")
    else:
        triggers = stim._triggers
    if stim._channel_mask is None:
        channel_mask = h5py.Empty("f")
    else:
        channel_mask = stim._channel_mask
    f.attrs["zoom"] = stim.zoom
    f.attrs["label"] = label
    f.create_dataset(
        "triggers",
        data=triggers,
        dtype="uint64",
        **filter_opts(triggers, dataset_opts | chunk_opts(triggers)),
    )
    f.create_dataset(
        "channel_mask",
        data=channel_mask,
        dtype="uint8",
        **filter_opts(channel_mask, dataset_opts | chunk_opts(channel_mask)),
    )
    f.create_dataset(
        "frames",
        data=stim._frames,
        dtype="uint8",
        **filter_opts(stim._frames, dataset_opts | chunk_opts(stim._frames)),
    )
    # HDF5 supports 0-dim datasets, so we can store the float|Sequence[float] directly.
    f.create_dataset(
        "frame_durations",
        data=stim._frame_durations,
        dtype="float64",
        **filter_opts(
            stim._frame_durations, dataset_opts | chunk_opts(stim._frame_durations)
        ),
    )

    # Always create metadata group and store attributes
    metadata_group = f.create_group("metadata")
    for key, value in stim.metadata.items():
        metadata_group.attrs[key] = value


def _get_attr(f, key, default=None):
    """Get hdf5 value from key, and use default if the value is h5pf.Empty.

    According to docs,

        https://docs.h5py.org/en/latest/high/dataset.html#creating-and-reading-empty-or-null-datasets-and-attributes

    empty/missing values can be stored as h5py.Empty objects. The best way to check for
    such empty values is to test the objects "shape" attribute and check if it is none.
    """
    res = f.attrs.get(key, None)
    # This doesn't work well for me. Non empty types error.
    if res is None:
        res = default
    return res


def get_dataset(f, key, default=None):
    """Get hdf5 dataset from key, and use default if the dataset is h5pf.Empty."""
    res = f.get(key, None)
    if res is not None:
        res = res[()]
        if isinstance(res, h5py.Empty):
            res = None
    if res is None:
        res = default
    return res


def _read_hdf5_v1(f):
    """Load v1 format.

    This is the current (experimental) format.

    The presentation currently only supports presenting 8-bit sRGB images, and if the
    frame dtype is not uint8, it will be converted to uint8.
    """
    ver = f.attrs["format_version"]
    if ver != "1":
        _logger.warning(f"Expected format version 1, got {ver}")
    zoom = _get_attr(f, "zoom", default=1)
    # Keep frames as an h5py.Dataset (lazy). _write_hdf5_v1 always writes uint8
    # (see line ~853), so this should already be uint8 for any file produced by
    # this codebase. If a foreign file isn't uint8, materialize and convert
    # eagerly (slow, but rare).
    frames = f["frames"]
    frame_durations = f["frame_durations"][()]
    triggers = get_dataset(f, "triggers", default=None)
    channel_mask = get_dataset(f, "channel_mask", default=None)
    label = _get_attr(f, "label")
    metadata = dict(f["metadata"])
    if frames.dtype != np.uint8:
        _logger.warning(
            f"Expected uint8 dtype, got {frames.dtype=}. Materializing and "
            f"converting to uint8 (load will be slow for large stims)."
        )
        frames = frames[()].astype(np.uint8)

    return StimArray(
        frames=frames,
        frame_durations=frame_durations,
        zoom=zoom,
        triggers=triggers,
        channel_mask=channel_mask,
        label=label,
        metadata=metadata,
    )


def _read_hdf5_v0(f):
    """Load the original format."""
    stim = np.asarray(f["Noise"][:], dtype=np.uint8)
    frame_rate = f["Frame_Rate"][()]
    print(stim.shape)
    if stim.ndim == 1:
        stim = einops.rearrange(stim, "f -> f 1 1 1")
    elif stim.ndim == 2:
        stim = einops.rearrange(stim, "f c -> f 1 1 c")
    elif stim.ndim == 3:
        stim = einops.rearrange(stim, "f h w -> f h w 1")
    print(stim.shape)
    frame_duration = 1.0 / frame_rate
    res = StimArray(
        frames=stim, frame_durations=frame_duration, zoom=1, triggers=None, label=None
    )
    return res


def _preview_hdf5_v0(f):
    """Preview the original format."""
    n_frames, h, w = f["Noise"][:].shape[0:3]
    fps = f["Frame_Rate"][()]
    stim_duration = n_frames * (1.0 / fps)
    checkerboard_size = f["Checkerboard_Size"][()]
    shuffle = f["Shuffle"][()]
    metadata = {"checkerboard_size": checkerboard_size, "shuffle": shuffle}
    return {
        "n_frames": n_frames,
        "height": h,
        "width": w,
        "fps": fps,
        "duration": stim_duration,
        "metadata": metadata,
    }


def _preview_hdf5_v1(f):
    """Preview the v1."""
    # Convention is to always have 4 dims.
    N, F, H, W, C = stim_shape_from_hdf5_v1(f)
    triggers = f.get("triggers", None)
    # "triggers" can be missing, or point to dataset or h5py.Empty.
    if triggers is None:
        n_triggers = 0
    elif triggers.shape is None:
        # For h5py.Empty, shape is None.
        # But we may have set it to an empty array, so checking for instance(h5py.Empty)
        # won't catch that case.
        n_triggers = 0
    else:
        n_triggers = triggers.shape[0]
    channel_mask = f.get("channel_mask", None)
    channel_mask_shape = channel_mask.shape if channel_mask is not None else None
    zoom = _get_attr(f, "zoom", default=1)
    metadata = dict(f["metadata"])
    frame_times = frame_times_from_hdf5_v1(f)
    duration = frame_times[-1]
    if f["frame_durations"].ndim == 0:
        spf = f["frame_durations"][()]
        fps = 1.0 / spf
    else:
        fps = None

    return {
        "n_frames": F,
        "n_mask_repeats": N,
        "n_total_frames": N * F,
        "n_triggers": n_triggers,
        "channel_mask.shape": channel_mask_shape,
        "fps": fps,
        "duration": duration,  # seconds
        "height": H,
        "width": W,
        "channels": C,
        "zoom": zoom,
        "metadata": metadata,
    }


def frame_times_from_hdf5_v0(f) -> np.ndarray:
    """Get frame times from v0 format HDF5 file."""
    n_frames = f["Noise"][:].shape[0]
    fps = f["Frame_Rate"][()]
    spf = 1.0 / fps
    frame_times = spf_to_frame_times(spf, n_frames)
    return frame_times


def stim_shape_from_hdf5_v1(f) -> Tuple[int, int, int, int, int]:
    """Get stimulus shape from v1 format HDF5 file."""
    F, H, W, C = f["frames"][:].shape
    channel_mask = f.get("channel_mask", None)
    if channel_mask is not None and channel_mask.ndim == 3:
        N = channel_mask.shape[0]
    else:
        N = 1
    return N, F, H, W, C


def frame_times_from_hdf5_v1(f) -> np.ndarray:
    """Get frame times from v1 format HDF5 file."""
    frame_durations = f["frame_durations"]
    if frame_durations.ndim not in (0, 1, 2):
        raise ValueError(
            f"Expected frame_durations to be 0D, 1D or 2D. ({frame_durations.ndim=})"
        )
    N, F, H, W, C = stim_shape_from_hdf5_v1(f)
    frame_durations = frame_durations[()]
    frame_times = _frame_times(frame_durations, N, F)
    return frame_times


class StimProgram(Protocol):
    """Knows how to play each frame of a stimulus.

    Called by each Presenter instance in their presentation loop.


    Using typing's `Protocol` improves type checking and IDE support, but otherwise has
    no effect.
    For usage of Python's "..." see https://github.com/python/typing/issues/109
    """

    def setup(
        self,
        ctx: moderngl.Context,
        win_width,
        win_height,
        channels: Optional[Sequence[int]],
        mirror: Optional[bool],
        rotation: Optional[float],
        win_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the stimulus program, returning the triggers array if any.

        Returns the frame times and triggers (two numpy arrays).

        Frame times is a 1D array of length num_frames + 1 (includes the end time).

        Mirror and rotate describe how the window has been configured differently than
        the stimulus expects by default. The stimulus program may optionally use this
        information to adjust rendering.

        Parameters
        ----------
        ctx : moderngl.Context
            The OpenGL context.
        win_width : int
            The width of the window in pixels.
        win_height : int
            The height of the window in pixels.
        channels : Optional[Sequence[int]]
            The channels the window wants to display.
        mirror : bool
            Whether to mirror the stimulus horizontally.
        rotation : int
            The rotation (degrees) to apply to the stimulus (after any mirror).
        """
        ...

    def is_linear(self) -> bool:
        """Return True if values are linear RGB, or False if sRGB.

        It is assumed that the display will received sRGB values
        """
        ...

    def render(self, ctx, frame_idx, global_frame_num) -> None: ...

    def cleanup(self) -> None: ...


# Just use from_script and from_hdf5. No need for protocol.
# class StimGenerator(Protocol):
#     """Knows how to instantiate a StimPogram.
#
#     Maybe we will only every need one implementation of this, as we have gone with the
#     most powerful: load an arbitrary python module. Keeping this interface class here as
#     a nod to all the other possible approaches, such as making a class per stimulus
#     type, such as CheckerboardStim, ChirpStim, etc.
#     """
#
#     def to_program(self) -> StimProgram: ...


def from_script(module_path: Path, config: Optional[str]) -> StimProgram:
    """Create a StimProgram from a script module.

    The module is expected to have the following function:

        * to_program(config: Optional[str]) -> StimProgram
        * default_config() -> str     [optional function, for display in GUI]

    Instead of having to serialize your stimulus to a StimArray as a hdf5 file, you can
    instead define a script programmatically, which serves each frame on demand.
    See examples/resources/example_program.py for an example of how to write such a
    script. Writing a program this way can save a lot of disk space, and it can allow
    you to leverage OpenGL for the stimulus generation. Moving bars is an example where
    the OpenGL shader is relatively easy, but to create numpy arrays by hand is harder
    due to having to manually account for aliasing and partially covered pixels.

    It's up to the script how to interpret the config string. JSON makes sense for many
    cases.
    """
    module_path = Path(module_path)

    if not module_path.exists():
        raise FileNotFoundError(f"Script not found: {module_path}")

    # Load module from file path using importlib.util
    spec = importlib.util.spec_from_file_location(
        module_path.stem, module_path  # Use filename as module name
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "to_program"):
        raise ValueError(
            f"Module {module_path} does not have required function to_program()."
        )

    return module.to_program(config)


def create_program(path: Path, config: Optional[str] = None) -> StimProgram:
    if path.suffix in (".h5", ".hdf5"):
        stim_array = StimArray.read_hdf5(path)
        lazy_textures = False
        if config is not None:
            config_dict = json.loads(config)
            lazy_textures = config_dict.get("lazy_textures", False)
        stim_program = TextureSequence(stim_arr=stim_array, lazy_textures=lazy_textures)
    elif path.suffix in MOVIE_SUFFIXES:
        stim_program = MoviePlayer(movie_path=path)
    elif path.suffix == ".py":
        stim_program = from_script(path, config)
    else:
        raise ValueError(f"Unsupported stimulus file type: {path.suffix}")
    return stim_program


class TextureSequence(StimProgram):
    """Render numpy arrays (from a StimArray) as a sequence of OpenGL textures."""

    # We always bind to texture unit 0.
    TEXTURE_UNIT = 0

    def __init__(self, stim_arr: StimArray, lazy_textures=False) -> None:
        self.stim_arr = stim_arr
        self.lazy_textures = lazy_textures
        self.textures = None
        self.single_tex = None
        self.win_id = None
        self._setup_done = False

    def setup(
        self,
        ctx,
        win_width,
        win_height,
        channels=None,
        mirror=None,
        rotation=None,
        win_id=None,
    ):
        if self._setup_done:
            raise RuntimeError("TextureSequence.setup() has already been called.")
        self.win_id = win_id

        if channels is not None:
            self.stim_arr = self.stim_arr.with_channels(channels)
            # now we just keep the whole thing, and index it correctly when needed.
            pass
        if mirror is None:
            mirror = False
        if rotation is None:
            rotation = 0

        # Load textures
        N, F, H, W, C_all = self.stim_arr.shape
        C = len(channels) if channels is not None else C_all
        if C > 4:
            raise ValueError(f"Too many channels requested. {C=}. Max for GLSL 4.")
        if not self.lazy_textures:
            # Create textures for all frames.
            self.textures = []
            frames = einops.rearrange(
                self.stim_arr.masked_frames(), "n f h w c -> (n f) h w c"
            )
            # No CPU-side V flip: the vertex shader inverts uv.y so texture (0,0)
            # samples to the top of the quad, matching StimArray's
            # (0,0)=top-left convention.

            for i in range(F * N):
                # masked_frame = self.stim_arr.frame_at(i)
                masked_frame = frames[i]
                assert masked_frame.shape == (H, W, C)
                _logger.info(f"Load frame:\t {i} \t{masked_frame.shape=}")
                tex = ctx.texture(
                    (W, H), C, masked_frame.tobytes(), samples=0, alignment=1
                )
                # No filtering. Expect aliasing.
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self.textures.append(tex)
            assert len(self.textures) == N * F, f"{N=}, {F=}, {len(self.textures)=}"
        # Compile program and load vertices.
        self._program = self._compile_program(ctx)
        zoom = self.stim_arr.zoom
        quad = create_centered_quad(
            W * zoom, H * zoom, win_width, win_height, mirror, rotation
        )
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.simple_vertex_array(self._program, self._vbo, "in_pos")
        self._setup_done = True
        frame_times, triggers = self.stim_arr.frame_times(), self.stim_arr.triggers()
        return frame_times, triggers

    def _compile_program(self, ctx):
        """Read shaders and compile the program."""
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / QUAD_VERTEX_SHADER).open("r") as vertex_file:
            vert_src = vertex_file.read()
        with (resource_dir / TEXTURE_FRAG_SHADER).open("r") as fragment_file:
            frag_src = fragment_file.read()
        program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        program["tex"].value = self.TEXTURE_UNIT
        return program

    def render(self, ctx, frame_idx, global_frame_num):
        """
        Design note: we could pass global_frame_num only, and have this class decide
        what to do. That would make it have to know about looping. Or, we could only
        pass frame_idx, assumed to be looped, and don't pass global_frame_num at all.
        """
        # TODO: zoom!
        if self.lazy_textures:
            N, F, H, W, C_all = self.stim_arr.shape
            C = 3  # always RGB.
            assert frame_idx < N * F, f"Index out of bounds. {frame_idx=}, {N=}, {F=}"
            if self.single_tex is None:
                self.single_tex = ctx.texture(
                    (W, H),
                    C,
                    samples=0,
                    alignment=1,
                )
                self.single_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            frame = self.stim_arr.frame_at(frame_idx)
            # No CPU-side V flip needed: the vertex shader inverts uv.y.
            self.single_tex.write(frame.tobytes())
            self.single_tex.use(location=self.TEXTURE_UNIT)
        else:
            assert self.textures is not None
            self.textures[frame_idx].use(location=self.TEXTURE_UNIT)
        self._vao.render(moderngl.TRIANGLES)

    def cleanup(self) -> None:
        """Release OpenGL resources and any HDF5 file held by the StimArray."""
        if self.single_tex is not None:
            self.single_tex.release()
            self.single_tex = None
        if self.textures is not None:
            for tex in self.textures:
                tex.release()
            self.textures = None
        if self._vao is not None:
            self._vao.release()
        if self._vbo is not None:
            self._vbo.release()
        self._vao = None
        self._vbo = None
        if self.stim_arr is not None:
            self.stim_arr.close()


class ProceduralShader(StimProgram):
    """A procedural shader-based stimulus.

    This class represents an arbitrary shader with its uniforms. The shader is
    loaded from a file, and uniforms are set from a dictionary.

    This class is not a core part of fpspy, and could be moved to an example directory.
    """

    def __init__(
        self,
        frag_shader_path: Path,
        width: int,
        height: int,
        uniforms: dict,
        # Proposed interface. Need to have enough info to return frame times and triggers.
        n_frames: int,
        spf: float,
        triggers: np.ndarray,
        label: Optional[str] = None,
    ):
        # Assume all programs use the same vertex shader (basic quad).
        self.frag_shader_path = frag_shader_path
        self.width = width
        self.height = height
        self.uniforms = uniforms
        self.n_frames = n_frames
        self.spf = spf
        self.triggers = triggers
        self.label = label

        # Set during init().
        self.win_id = None
        self._program = None
        self._vao = None
        self._vbo = None

    def setup(
        self,
        ctx: moderngl.Context,
        win_width,
        win_height,
        channels: Optional[Sequence[int]],
        mirror: Optional[bool] = None,
        rotation: Optional[float] = None,
        win_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the shader program."""
        self.win_id = win_id
        if mirror is None:
            mirror = False
        if rotation is None:
            rotation = 0

        # Load shaders and compile the program.
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / "vertex_shader.glsl").open("r") as vert_f:
            vert_src = vert_f.read()
        with open(self.frag_shader_path, "r") as frag_f:
            frag_src = frag_f.read()
        self._program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Set static uniforms from the uniforms dict.
        _set_uniforms(self._program, self.uniforms)
        # Set mirror/rotation uniforms for UV coordinate transformation.
        self._program["u_mirror"].value = mirror
        self._program["u_rotation"].value = float(rotation)

        quad = create_centered_quad(self.width, self.height, win_width, win_height)
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.simple_vertex_array(self._program, self._vbo, "in_pos")
        frame_times = spf_to_frame_times(self.spf, self.n_frames)
        return frame_times, self.triggers

    def render(self, ctx, frame_idx, global_frame_num) -> None:
        """Render a single frame.

        Parameters
        ----------
        ctx : moderngl.Context
            The OpenGL context (unused but required by protocol).
        frame_idx : int
            The frame index within the stimulus.
        global_frame_num : int
            The global frame number (passed to shader as uniform).
        """
        self._program["frame_num"].value = frame_idx
        self._vao.render(moderngl.TRIANGLES)

    def cleanup(self):
        """Release OpenGL resources."""
        if self._vao is not None:
            self._vao.release()
        if self._vbo is not None:
            self._vbo.release()
        self._vao = None
        self._vbo = None


class MoviePlayer(StimProgram):
    """Play a video file as a stimulus.

    Uses PyAV (libav) to decode on the CPU and uploads each frame via
    `texture.write`. Trades a small per-frame decode cost for much smaller
    files on disk (temporal compression) and the ability to read fps directly
    from the container.

    Frame rate comes from the video stream's `average_rate`; triggers default
    to one per frame. Variable-frame-rate inputs are rejected at setup.
    """

    TEXTURE_UNIT = 0

    def __init__(
        self,
        movie_path: Path,
        zoom: float = 1.0,
        is_linear: bool = False,
        label: Optional[str] = None,
    ) -> None:
        self.movie_path = Path(movie_path)
        self.zoom = zoom
        self._is_linear = is_linear
        self.label = label if label is not None else self.movie_path.stem

        self.win_id = None
        self._container = None
        self._stream = None
        self._decoder_iter = None
        self._last_decoded_idx = -1
        self._last_frame_bytes = None
        self._n_frames = None
        self._width = None
        self._height = None
        self._tex = None
        self._program = None
        self._vbo = None
        self._vao = None

    def setup(
        self,
        ctx: moderngl.Context,
        win_width,
        win_height,
        channels: Optional[Sequence[int]] = None,
        mirror: Optional[bool] = None,
        rotation: Optional[float] = None,
        win_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Defer PyAV import so the dep only matters when a movie is actually
        # played. Keeps `import fpspy.stim` fast and lets users skip the av
        # install if they never play movies.
        import av  # noqa: PLC0415

        self.win_id = win_id
        if mirror is None:
            mirror = False
        if rotation is None:
            rotation = 0
        if channels is not None:
            _logger.info(
                f"MoviePlayer ignores `channels`; video is rendered as-is. "
                f"{channels=}"
            )

        self._container = av.open(str(self.movie_path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        avg_rate = self._stream.average_rate
        guessed_rate = self._stream.guessed_rate
        if avg_rate is None or avg_rate != guessed_rate:
            raise ValueError(
                f"Movie {self.movie_path} appears to be variable-frame-rate "
                f"({avg_rate=}, {guessed_rate=}). Re-encode with constant "
                f"frame rate (e.g. `ffmpeg -r 30 -vsync cfr`)."
            )
        n_frames = self._stream.frames
        if n_frames <= 0:
            raise ValueError(
                f"Movie {self.movie_path} reports {n_frames} frames; cannot "
                f"compute frame times. Re-mux with a tool that writes frame "
                f"counts (e.g. `ffmpeg -i in.mp4 -c copy out.mp4`)."
            )
        spf = 1.0 / float(avg_rate)
        self._n_frames = n_frames
        self._width = self._stream.width
        self._height = self._stream.height

        self._tex = ctx.texture(
            (self._width, self._height), 3, samples=0, alignment=1
        )
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self._program = self._compile_program(ctx)
        quad = create_centered_quad(
            int(self._width * self.zoom),
            int(self._height * self.zoom),
            win_width,
            win_height,
            mirror,
            rotation,
        )
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.simple_vertex_array(self._program, self._vbo, "in_pos")

        self._decoder_iter = self._make_iter()

        frame_times = spf_to_frame_times(spf, n_frames)
        triggers = np.arange(n_frames, dtype=np.uint64)
        return frame_times, triggers

    def _compile_program(self, ctx):
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / QUAD_VERTEX_SHADER).open("r") as f:
            vert_src = f.read()
        with (resource_dir / TEXTURE_FRAG_SHADER).open("r") as f:
            frag_src = f.read()
        program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        program["tex"].value = self.TEXTURE_UNIT
        return program

    def _make_iter(self):
        # rgb24 matches the 3-channel RGB texture; PyAV does YUV->RGB via
        # libswscale on the CPU.
        return (
            frame.to_ndarray(format="rgb24")
            for frame in self._container.decode(video=0)
        )

    def render(self, ctx, frame_idx, global_frame_num) -> None:
        # Video decoders are sequential. Three cases:
        #   * frame_idx == last  -> reuse the texture (display fps > stim fps).
        #   * frame_idx >  last  -> advance the iterator that many steps.
        #   * frame_idx <  last  -> loop wrap; seek to start and re-decode.
        assert frame_idx < self._n_frames, (
            f"Frame index out of bounds. {frame_idx=}, {self._n_frames=}"
        )
        if frame_idx < self._last_decoded_idx:
            self._container.seek(0, stream=self._stream)
            self._decoder_iter = self._make_iter()
            self._last_decoded_idx = -1
        if frame_idx > self._last_decoded_idx:
            frame = None
            while self._last_decoded_idx < frame_idx:
                frame = next(self._decoder_iter)
                self._last_decoded_idx += 1
            assert frame is not None
            self._tex.write(frame.tobytes())
        self._tex.use(location=self.TEXTURE_UNIT)
        self._vao.render(moderngl.TRIANGLES)

    def is_linear(self) -> bool:
        return self._is_linear

    def cleanup(self) -> None:
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        if self._container is not None:
            self._container.close()
            self._container = None
