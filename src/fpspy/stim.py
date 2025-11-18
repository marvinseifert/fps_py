from pathlib import Path
import h5py
import warnings

__all__ = ["Stim"]

CURRENT_HDF5_FORMAT_VER = "1"


class Stim:
    """A class representing a stimulus.

    Currently, the main purpose is to encapsulate the serialization and deserialization.
    """

    def __init__(self, frames, fps, triggers=None, label=None):
        """
        Parameters
        ----------
        frames : np.ndarray
            3D numpy array with shape (num_frames, height, width, channels).
        fps : int
            Frames per second of the stimulus.
        label : str, optional
            An optional label for the stimulus.
        """
        self.frames = frames
        self.fps = fps
        self.triggers = triggers
        self.label = label

    def n_frames(self):
        return self.frames.shape[0]

    def height(self):
        return self.frames.shape[1]

    def width(self):
        return self.frames.shape[2]

    def n_channels(self):
        return self.frames.shape[3]

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
            ver = f.attrs.get("format_version", "1")
            if ver == "1":
                return _read_hdf5_v1(f)
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


def _read_hdf5_v1(f):
    ver = f.attrs["format_version"]
    if ver != "1":
        warnings.warn(f"Expected format version 1, got {ver}")
    fps = f.attrs["fps"]
    label = f.attrs["label"]
    frames = f["frames"][()]
    triggers = f["triggers"][()]
    if label.shape is None:
        label = None
    if triggers.shape is None:
        triggers = None
    return Stim(frames=frames, fps=fps, triggers=triggers, label=label)
