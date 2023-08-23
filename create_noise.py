import numpy as np
import h5py


# %%

def generate_checkerboard_pattern(checker_size, width_in_pixels, height_in_pixels):
    """
    Generate a checkerboard pattern with a given checker size and dimensions.

    Parameters
    ----------
    checker_size : int
        The size of the checkerboard squares in pixels.
    width_in_pixels : int
        The width of the pattern in pixels.
    height_in_pixels : int
        The height of the pattern in pixels.

    """
    # Calculate the number of squares based on pixel resolution
    pattern_width = width_in_pixels // checker_size
    pattern_height = height_in_pixels // checker_size

    pattern_shape = (pattern_width, pattern_height)
    pattern = np.random.randint(0, 2, pattern_shape, dtype=np.uint8) * 255
    pattern_texture = np.repeat(np.repeat(pattern, checker_size, axis=0), checker_size, axis=1)

    return pattern_texture



# %%
def generate_and_store_3d_array(frames, checkerboard_size, width_in_pixels, height_in_pixels, fps,name="Noise.h5"):
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

    patterns_list = [generate_checkerboard_pattern(checkerboard_size, width_in_pixels, height_in_pixels) for _ in
                     range(frames)]
    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array
    with h5py.File(name, 'w') as f:
        f.create_dataset('Noise', data=stacked_patterns, dtype="uint8")
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=checkerboard_size, dtype="uint64")
        f.create_dataset(name="Shuffle", data=False, dtype="bool")



