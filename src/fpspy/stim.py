import logging
import time
from pathlib import Path
from typing import Protocol, Optional, Sequence, Tuple
import importlib.resources
import h5py
import numpy as np
import einops
import moderngl
import importlib
import importlib.util


_logger = logging.getLogger(__name__)

__all__ = [
    "StimArray",
    "TextureSequence",
    "ProceduralShader",
    "StimProgram",
    "CURRENT_HDF5_FORMAT_VER",
]


CURRENT_HDF5_FORMAT_VER = "1"
TEXTURE_FRAG_SHADER = "fragment_shader_colour.glsl"
QUAD_VERTEX_SHADER = "vertex_shader.glsl"


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
    trigger_repeats = [triggers]
    for i in range(1, n_loops):
        trigger_repeats.append(triggers + i * n_frames)
    triggers_out = np.concatenate(trigger_repeats)
    triggers_out = triggers_out.astype(int)
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


def create_centered_quad(stim_width, stim_height, win_width, win_height):
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


class StimArray:
    """An array-based stimulus.

    Currently, the main purpose is to encapsulate the serialization and deserialization.
    """

    def __init__(
        self,
        frames,  # f, h, w, 1
        frame_times: float | Sequence[float],
        zoom,
        triggers=None,
        label=None,
        metadata=None,
    ):
        """
        Parameters
        ----------
        frames : np.ndarray
            3D numpy array with shape (num_frames, height, width, channels).
        triggers : np.ndarray, optional
            1D numpy array of trigger times (in frame indices). If None, triggers
        frame_times : float | Sequence[float]
            Either a single float representing a shared frame duration (seconds), or a
            sequence of frame times for each frame (seconds). If a sequence is provided,
            its length must ben len(framems)+1, and the first element must be 0. The
            last element represents the end time of the last frame.
        zoom: int >= 1
            Zoom factor for displaying the stimulus.
        label : str, optional
            An optional label for the stimulus.
        metadata : dict, optional
            An optional dictionary of metadata. Things like checkerboard size
            can go in here, which may not be applicable to all stimuli.
            The metadata should be a non-nested dictionary with key-value pairs
            that are serializable with hdf5.
        """
        if frames.ndim != 4:
            raise ValueError(f"Expected shape (f, h, w, c). Got {frames.shape=}.")
        self.frames = frames
        self._set_frame_times(frame_times)
        if zoom < 1:
            raise ValueError(f"Zoom must be >= 1. Got {zoom=}.")
        #  check not integer
        if int(zoom) != zoom:
            raise ValueError(f"Zoom must be integer. Got {zoom=}.")
        self.zoom = int(zoom)
        if triggers is not None:
            if triggers.ndim != 1:
                raise ValueError(f"Expected 1D triggers. Got {triggers.shape=}.")
            if len(triggers) == 0:
                # Defensive. Having empty arrays is error prone as testing triggers!=None
                # doesn't guarantee we have triggers.
                raise ValueError("Triggers cannot be empty array. Use None instead.")
        self._triggers = triggers
        self.label = label
        self.metadata = metadata if metadata is not None else {}

    def __len__(self):
        return len(self.frames)

    @property
    def height(self):
        return self.frames.shape[1]

    @property
    def width(self):
        return self.frames.shape[2]

    @property
    def n_channels(self):
        nch = self.frames.shape[3]
        return nch

    def _set_frame_times(self, frame_times):
        self._frame_times = frame_times
        if not self._frame_times_is_scalar():
            # Validate length
            if not len(self.frame_times()) == len(self.frames) + 1:
                raise ValueError(
                    f"Expected len(frame_times) == len(frames)+1. "
                    f"Got {len(self.frame_times())=}, {len(self.frames)=}."
                )
            # Convert to numpy array
            self._frame_times = np.asarray(self._frame_times, dtype=np.float64)

    def _frame_times_is_scalar(self):
        """Return True if frame_times is a single float (fps)."""
        res = np.ndim(self._frame_times) == 0
        return res

    def fps(self, allow_estimate: bool):
        """Return the stimulus fps, if frame times are evenly spaced, None otherwise."""
        res = None
        if self._frame_times_is_scalar():
            res = 1.0 / self._frame_times
        else:
            if allow_estimate:
                # Are all frame times evenly spaced?
                diffs = np.diff(self._frame_times)
                if np.allclose(diffs, diffs[0]):
                    res = 1.0 / diffs[0]
        return res

    def frame_times(self) -> np.ndarray:
        if self._frame_times_is_scalar():
            spf = self._frame_times
            res = spf_to_frame_times(spf, len(self.frames))
        else:
            res = self._frame_times
        return res

    def frame_start_times(self) -> np.ndarray:
        start_and_final = self.frame_times()
        return start_and_final[:-1]

    def triggers(self):
        if self._triggers is None:
            triggers = np.arange(len(self.frames), dtype=np.uint64)
        else:
            triggers = self._triggers
        return triggers

    def __repr__(self):
        tr_str = f"len(triggers)={len(self._triggers)}" if self._triggers else "None"
        return (
            f"Stim(shape(frames)={self.frames.shape}, "
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
        # If not always 4 dim, then use:
        # new_frames = self.frames[..., channels]
        print(self.frames.shape)
        new_frames = self.frames[:, :, :, channels]
        return StimArray(
            frames=new_frames,
            frame_times=self._frame_times,
            zoom=self.zoom,
            triggers=self._triggers,
            label=self.label,
            metadata=self.metadata,
        )

    def write_hdf5(self, path: Path):
        """Write the stimulus to an HDF5 file.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file where the stimulus metadata will be stored.
        """

        with h5py.File(path, "w") as f:
            _write_hdf5_v1(self, f)

    @staticmethod
    def read_hdf5(path: Path):
        """Read a stimulus from an HDF5 file.

        Parameters
        ----------
        path : Path
            The path to the HDF5 file from which to read the stimulus metadata.

        Returns
        -------
        Stim
            The stimulus object created from the HDF5 file.
        """
        with h5py.File(path, "r") as f:
            ver = f.attrs.get("format_version", "0")
            if ver == "0":
                return _read_hdf5_v0(f)
            elif ver == "1":
                return _read_hdf5_v1(f)
            else:
                raise ValueError(f"Unsupported HDF5 format version: {ver}")

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


def _write_hdf5_v1(stim: StimArray, f):
    f.attrs["format_version"] = "1"
    label = stim.label if stim.label is not None else h5py.Empty("f")
    if stim._triggers is None or len(stim._triggers) == 0:
        triggers = h5py.Empty("f")
    else:
        triggers = stim._triggers
    f.attrs["zoom"] = stim.zoom
    f.attrs["label"] = label
    f.create_dataset("triggers", data=triggers, dtype="uint64")
    f.create_dataset("frames", data=stim.frames, dtype="uint8")
    # HDF5 supports 0-dim datasets, so we can store the float|Sequence[float] directly.
    f.create_dataset("frame_times", data=stim._frame_times, dtype="float64")

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
    frames = f["frames"][()]
    frame_times = f["frame_times"][()]
    triggers = get_dataset(f, "triggers", default=None)
    label = _get_attr(f, "label")
    metadata = dict(f["metadata"])
    if frames.dtype != np.uint8:
        _logger.warning(
            f"Expected uint8 dtype, got {frames.dtype=}. Converting to uint8."
        )
        frames = frames.astype(np.uint8)

    return StimArray(
        frames=frames,
        frame_times=frame_times,
        zoom=zoom,
        triggers=triggers,
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
        frames=stim, frame_times=frame_duration, zoom=1, triggers=None, label=None
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
    n_frames, h, w, c = f["frames"][:].shape
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
        print(triggers)
        n_triggers = triggers.shape[0]
    zoom = _get_attr(f, "zoom", default=1)
    metadata = dict(f["metadata"])
    if f["frame_times"].ndim == 0:
        spf = f["frame_times"][()]
        fps = 1.0 / spf
        duration = n_frames * spf
    else:
        assert f["frame_times"].ndim == 1
        frame_times = f["frame_times"][()]
        duration = frame_times[-1]
        fps = None

    return {
        "n_frames": n_frames,
        "n_triggers": n_triggers,
        "fps": fps,
        "duration": duration,  # seconds
        "height": h,
        "width": w,
        "channels": c,
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


def frame_times_from_hdf5_v1(f) -> np.ndarray:
    """Get frame times from v1 format HDF5 file."""
    frame_times = f["frame_times"]
    if frame_times.ndim not in (0, 1):
        raise ValueError(
            f"Expected frame_times to be 0D or 1D. Got {frame_times.ndim}D."
        )
    if frame_times.ndim == 0:
        spf = frame_times[()]
        n_frames = f["frames"].shape[0]
        frame_times = spf_to_frame_times(spf, n_frames)
    else:
        frame_times = frame_times[()]
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

    It's totally up to the script how to interpret the config string. JSON makes sense
    for many cases.
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
        stim_program = TextureSequence(stim_arr=stim_array, lazy_textures=False)
    elif path.suffix == ".py":
        stim_program = from_script(path, config)
    else:
        raise ValueError(f"Unsupported stimulus file type: {path.suffix}")
    return stim_program


class TextureSequence(StimProgram):

    # We always bind to texture unit 0.
    TEXTURE_UNIT = 0

    def __init__(self, stim_arr: StimArray, lazy_textures=False) -> None:
        self.stim_arr = stim_arr
        self.lazy_textures = lazy_textures
        self.textures = None
        self.single_tex = None
        self.win_id = None
        self._setup_done = False

    def setup(self, ctx, win_width, win_height, channels=None, mirror=None,
              rotation=None, win_id=None):
        if self._setup_done:
            raise RuntimeError("TextureSequence.setup() has already been called.")
        self.win_id = win_id

        if channels is not None:
            self.stim_arr = self.stim_arr.with_channels(channels)
        if mirror is None:
            mirror = False
        if rotation is None:
            rotation = 0

        # Load textures
        F, H, W, C = self.stim_arr.frames.shape
        if not self.lazy_textures:
            self.textures = []
            for i in range(F):
                tex = ctx.texture(
                    (W, H), C, self.stim_arr.frames[i].tobytes(), samples=0, alignment=1
                )
                # Use nearest-neighbor filtering for crisp pixels when zoomed.
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self.textures.append(tex)
        # Compile program and load vertices.
        self._program = self._compile_program(ctx)
        # Set mirror/rotation uniforms for UV coordinate transformation.
        self._program["u_mirror"].value = mirror
        self._program["u_rotation"].value = float(rotation)
        zoom = self.stim_arr.zoom
        quad = create_centered_quad(W * zoom, H * zoom, win_width, win_height)
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
            F, H, W, C = self.stim_arr.frames.shape
            # Create texture on demand.
            assert frame_idx < len(self.stim_arr)
            if self.single_tex is not None:
                self.single_tex.release()
            self.single_tex = ctx.texture(
                (W, H),
                C,
                self.stim_arr.frames[frame_idx].tobytes(),
                samples=0,
                alignment=1,
            )
            # Use nearest-neighbor filtering for crisp pixels when zoomed.
            self.single_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.single_tex.use(location=self.TEXTURE_UNIT)
        else:
            assert self.textures is not None
            self.textures[frame_idx].use(location=self.TEXTURE_UNIT)
        self._vao.render(moderngl.TRIANGLES)

    def cleanup(self) -> None:
        """Release OpenGL resources."""
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
