import numpy as np
import h5py
from create_noise import generate_checkerboard_pattern
import hdf5plugin
import blosc


def shuffle_pattern(pattern, checker_size):
    """Shuffle the pattern by a random number of pixels relative to checkerboard size in x and y directions.
    Parameters
    ----------
    pattern : numpy.ndarray
        The pattern to shuffle.
    checker_size : int
        The size of the checkerboard squares in pixels.
    Returns
    -------
    numpy.ndarray
        The shuffled pattern.

    """
    max_shift = int(checker_size - checker_size/4)  # Calculate maximum shift relative to checker size
    shifts = np.arange(0, max_shift + 1, checker_size // 4)  # Get the possible shifts


    # Generate random shifts for x and y from the calculated shifts
    x_shift = np.random.choice(shifts)
    y_shift = np.random.choice(shifts)

    # Use numpy's roll function to perform the shifts
    shifted_pattern = np.roll(pattern, shift=x_shift, axis=1)  # Shift in x
    shifted_pattern = np.roll(shifted_pattern, shift=y_shift, axis=0)  # Shift in y

    return shifted_pattern


def generate_and_store_3d_array(frames, checkerboard_size, width_in_pixels, height_in_pixels, fps, name="Noise.h5"):
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
    name : str
        The name of the HDF5 file to store the pattern in.
    """

    patterns_list = []

    # Generate the checkerboard patterns with random shuffling for each frame
    for _ in range(frames):
        pattern = generate_checkerboard_pattern(checkerboard_size, width_in_pixels, height_in_pixels)
        shuffled_pattern = shuffle_pattern(pattern, checkerboard_size)
        patterns_list.append(shuffled_pattern)

    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array

    with h5py.File(name, 'w') as f:
        f.create_dataset('Noise', data=stacked_patterns, dtype="uint8",
                         compression=hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=checkerboard_size, dtype="uint64")
        f.create_dataset(name="Shuffle", data=True, dtype="bool")

