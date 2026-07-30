from pathlib import Path
import h5py
import numpy as np
import fpspy.create_noise
import hdf5plugin


def valid_shuffle_steps(checker_size: int) -> list[int]:
    """The shuffle_step values valid for a given checker_size.

    Shift positions must start at 0 and be equally spaced (so the jump
    between any two adjacent positions is the same), and a shift can be at
    most checker_size - 2: at checker_size - 1, the neighbouring checker
    would cover this one by more of its width than its own, which defeats
    the point of a jittered-but-recognizable checkerboard. So the valid
    positions are {0, step, 2*step, ..., checker_size - 2}, which requires
    step to divide (checker_size - 2) evenly - otherwise the sequence either
    overshoots checker_size - 2 or stops short of it with uneven spacing.

    Returns the divisors of (checker_size - 2), smallest first. Empty if
    checker_size < 3 (no nonzero shift is valid at all).
    """
    max_shift = checker_size - 2
    if max_shift < 1:
        return []
    return [step for step in range(1, max_shift + 1) if max_shift % step == 0]


def shuffle_pattern(pattern, checker_size, shuffle_step):
    """Shuffle the pattern by a random number of pixels relative to checkerboard size in x and y directions.
    Parameters
    ----------
    pattern : numpy.ndarray
        The pattern to shuffle.
    checker_size : int
        The size of the checkerboard squares in pixels.
    shuffle_step : int
        Spacing, in pixels, between the possible shuffle positions. Must be
        one of `valid_shuffle_steps(checker_size)`.
    Returns
    -------
    numpy.ndarray
        The shuffled pattern.

    """
    max_shift = checker_size - 2
    if max_shift < 1 or max_shift % shuffle_step != 0:
        raise ValueError(
            f"shuffle_step={shuffle_step} is not valid for checker_size="
            f"{checker_size}. Valid steps: {valid_shuffle_steps(checker_size)}"
        )
    shifts = np.arange(0, max_shift + 1, shuffle_step)  # Get the possible shifts

    # Generate random shifts for x and y from the calculated shifts
    x_shift = int(np.random.choice(shifts))
    y_shift = int(np.random.choice(shifts))

    # Use numpy's roll function to perform the shifts
    shifted_pattern = np.roll(pattern, shift=x_shift, axis=1)  # Shift in x
    shifted_pattern = np.roll(shifted_pattern, shift=y_shift, axis=0)  # Shift in y

    return shifted_pattern


def generate_and_store_3d_array(
    frames: int,
    checkerboard_size: int,
    width_in_pixels: int,
    height_in_pixels: int,
    fps: int,
    shuffle_step: int,
    name: str | Path = "Noise.h5",
):
    """Generate a 3D array of checkerboard patterns and store it in an HDF5 file.
    Parameters
    ----------
    frames : int
        The number of frames to generate.
    checkerboard_size : int
        The size of the checkerboard squares in pixels.
    width_in_pixels : int
        The width of the pattern in pixels.
    height_in_pixels : int
        The height of the pattern in pixels.
    fps : int
        The frame rate of the pattern in Hz.
    shuffle_step : int
        Spacing, in pixels, between possible shuffle positions. Must be one
        of `valid_shuffle_steps(checkerboard_size)`.
    name : str
        The name of the HDF5 file to store the pattern in.
    """

    patterns_list = []

    # Generate the checkerboard patterns with random shuffling for each frame
    for _ in range(frames):
        pattern = fpspy.create_noise.checkerboard(
            checkerboard_size, width_in_pixels, height_in_pixels
        )
        shuffled_pattern = shuffle_pattern(pattern, checkerboard_size, shuffle_step)
        patterns_list.append(shuffled_pattern)

    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array

    with h5py.File(name, "w") as f:
        f.create_dataset(
            "Noise",
            data=stacked_patterns,
            dtype="uint8",
            compression=hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
            ),
        )
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(
            name="Checkerboard_Size", data=checkerboard_size, dtype="uint64"
        )
        f.create_dataset(name="Shuffle", data=True, dtype="bool")


def generate_and_store_3d_array_colour(
    frames,
    checkerboard_size,
    width_in_pixels,
    height_in_pixels,
    fps,
    num_channels,
    shuffle_step,
    name="Noise.h5",
):
    """Generate a 3D array of checkerboard patterns and store it in an HDF5 file.
    Parameters
    ----------
    frames : int
        The number of frames to generate.
    checkerboard_size : int
        The size of the checkerboard squares in pixels.
    width_in_pixels : int
        The width of the pattern in pixels.
    height_in_pixels : int
        The height of the pattern in pixels.
    fps : int
        The frame rate of the pattern in Hz.
    num_channels : int
        The number of channels in the pattern.
    shuffle_step : int
        Spacing, in pixels, between possible shuffle positions. Must be one
        of `valid_shuffle_steps(checkerboard_size)`.
    name : str
        The name of the HDF5 file to store the pattern in.
    """

    patterns_list = []

    # Generate the checkerboard patterns with random shuffling for each frame
    for _ in range(frames):
        pattern = fpspy.create_noise.multicolor_checkerboard(
            checkerboard_size,
            width_in_pixels,
            height_in_pixels,
            num_channels=num_channels,
        )
        shuffled_pattern = shuffle_pattern(pattern, checkerboard_size, shuffle_step)
        patterns_list.append(shuffled_pattern)

    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array

    with h5py.File(name, "w") as f:
        f.create_dataset(
            "Noise",
            data=stacked_patterns,
            dtype="uint8",
            compression=hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
            ),
        )
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(
            name="Checkerboard_Size", data=checkerboard_size, dtype="uint64"
        )
        f.create_dataset(name="Shuffle", data=True, dtype="bool")
