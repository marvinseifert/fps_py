import logging
from pathlib import Path
import h5py
import numpy as np
import einops

__all__ = ["Stim"]

CURRENT_HDF5_FORMAT_VER = "1"

_logger = logging.getLogger(__name__)


class Stim:
    """A class representing a stimulus.

    Currently, the main purpose is to encapsulate the serialization and deserialization.
    """

    def __init__(self, frames, fps, triggers=None, label=None, metadata=None):
        """
        Parameters
        ----------
        frames : np.ndarray
            3D numpy array with shape (num_frames, height, width, channels).
        fps : int
            Frames per second of the stimulus.
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
        self.fps = fps
        self.triggers = triggers
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

    def __repr__(self):
        tr_str = f"len(triggers)={len(self.triggers)}" if self.triggers else "None"
        return (
            f"Stim(shape(frames)={self.frames.shape}, fps={self.fps}, {tr_str},"
            f"label={self.label}, metadata={self.metadata.__repr__()}"
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
        new_frames = self.frames[:, :, :, channels]
        return Stim(
            frames=new_frames,
            fps=self.fps,
            triggers=self.triggers,
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
        with h5py.File(path, "r") as f:
            ver = f.attrs.get("format_version", "0")
            if ver == "0":
                return _preview_hdf5_v0(f)
            elif ver == "1":
                return _preview_hdf5_v1(f)
            else:
                raise ValueError(f"Unsupported HDF5 format version: {ver}")


def _write_hdf5_v1(stim: Stim, f):
    f.attrs["format_version"] = "1"
    label = stim.label if stim.label is not None else h5py.Empty("f")
    triggers = stim.triggers if stim.triggers is not None else h5py.Empty("f")
    f.attrs["fps"] = stim.fps
    f.attrs["label"] = label
    f.create_dataset("triggers", data=triggers, dtype="uint64")
    f.create_dataset("frames", data=stim.frames, dtype="uint8")

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
    # if ds.shape is None:
    if res is not None:
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
    fps = f.attrs["fps"][()]
    frames = f["frames"][()]
    triggers = _get_attr(f, "triggers")
    label = _get_attr(f, "label")
    metadata = dict(f["metadata"])
    if frames.dtype != np.uint8:
        _logger.warning(
            f"Expected uint8 dtype, got {frames.dtype=}. Converting to uint8."
        )
        frames = frames.astype(np.uint8)

    return Stim(
        frames=frames,
        fps=fps,
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
    res = Stim(frames=stim, fps=frame_rate, triggers=None, label=None)
    return res


def _preview_hdf5_v0(f):
    """Preview the original format."""
    n_frames, h, w = f["Noise"][:].shape[0:3]
    fps = f["Frame_Rate"][()]
    checkerboard_size = f["Checkerboard_Size"][()]
    shuffle = f["Shuffle"][()]
    metadata = {"checkerboard_size": checkerboard_size, "shuffle": shuffle}
    return {
        "n_frames": n_frames,
        "height": h,
        "width": w,
        "fps": fps,
        "metadata": metadata,
    }


def _preview_hdf5_v1(f):
    """Preview the v1."""
    # Convention is to always have 4 dims.
    n_frames, h, w, c = f["frames"][:].shape
    fps = f.attrs["fps"][()]
    metadata = dict(f["metadata"])
    return {
        "n_frames": n_frames,
        "height": h,
        "width": w,
        "channels": c,
        "fps": fps,
        "metadata": metadata,
    }
