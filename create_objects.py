import math
import numpy as np
import h5py
import hdf5plugin

def circle_path(t, radius=10, center=(15, 15)):
    """
    Define a circular path.

    Parameters:
        t: int, The time point.
        radius: int, The radius of the circle.
        center: tuple, The (x, y) coordinates of the circle's center.

    Returns:
        (x, y) coordinates
    """
    x_center, y_center = center
    # Convert t to radians as math.cos and math.sin expect radian arguments
    angle = math.radians(t)
    x = x_center + radius * math.cos(angle)
    y = y_center + radius * math.sin(angle)
    return int(x), int(y)


def linear_path(t, center, direction):
    """
    Define a linear path.

    Parameters:
        t: int, The time point.
        start_position: tuple, The (x, y) coordinates of the square's starting position.
        direction: tuple, The (dx, dy) direction of movement.

    Returns:
        (x, y) coordinates
    """
    x_start, y_start = 2*center[0], -center[1]
    dx, dy = direction

    # Calculate the new position based on the start position, direction, and time
    x = x_start + t * dx
    y = y_start + t * dy

    return int(x), int(y)




def create_square(duration, square_size, space_dim, path_func, name, **path_func_kwargs):
    """
    Create a 3D array representing a moving square in 2D space over time
    and save it as an HDF5 file.

    Parameters:
        duration: int, The number of time frames.
        space_dim: tuple, The (x, y) dimensions of the 2D space.
        path_func: function, Describes the path of the square. Takes time as input and returns (x, y).
        name: str, The name of the output HDF5 file.

    Returns:
        None
    """
    x_dim, y_dim = space_dim
    # Initialize a 3D array with all zeros
    space_time_matrix = np.zeros((duration, x_dim, y_dim), dtype=np.uint8)

    # Define the side length of the square


    center = (x_dim/2, y_dim/2)

    # Loop through each time point
    for t in range(duration):
        # Get the position of the square at time t
        x, y = path_func(t, center=center, **path_func_kwargs)

        # Ensure that the square is within bounds
        x = max(min(x, x_dim - square_size[1] - 1), 0)
        y = max(min(y, y_dim - square_size[0] - 1), 0)

        # Draw the square into the 2D space at time t
        space_time_matrix[t, x:x + square_size[1], y:y + square_size[0]] = 255

        # Save the 3D array to an HDF5 file with Blosc compression
    with h5py.File(name, 'w') as f:
        f.create_dataset('Noise', data=space_time_matrix, dtype="uint8",
                         compression=hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
        f.create_dataset(name="Frame_Rate", data=60, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=1, dtype="uint64")
        f.create_dataset(name="Shuffle", data=False, dtype="bool")


#create_square(20*60, 100, (500,500), circle_path, "stimuli/Moving_Circle.h5", radius=150)

create_square(20*60, (100, 500), (500,500), linear_path, "stimuli/Moving_Circle.h5", direction=(0,10))