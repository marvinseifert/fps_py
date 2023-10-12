import numpy as np

import matplotlib.pyplot as plt


def linear_path(t, center, direction, space_dim):
    """

    Define a linear path.



    Parameters:

        t: int, The time point.

        center: tuple, The (x, y) coordinates of the center of the space.

        direction: tuple, The (dx, dy) direction of movement.

        space_dim: tuple, The (x, y) dimensions of the 2D space.



    Returns:

        (x, y) coordinates

    """

    x_start, y_start = -center[0], center[1]

    dx, dy = direction

    x = x_start + t * dx

    y = y_start + t * dy

    return int(x), int(y)


def create_square(duration, bar_size, space_dim, path_func, direction):
    """

    Create a 3D array representing a moving bar in 2D space over time.



    Parameters:

        duration: int, The number of time frames.

        bar_size: tuple, The (x, y) dimensions of the bar.

        space_dim: tuple, The (x, y) dimensions of the 2D space.

        path_func: function, Describes the path of the bar. Takes time as input and returns (x, y).

        direction: tuple, The (dx, dy) direction of movement.



    Returns:

        space_time_matrix: ndarray, 3D array representing the moving bar over time.

    """

    x_dim, y_dim = space_dim

    space_time_matrix = np.zeros((duration, x_dim, y_dim), dtype=np.uint8)

    center = (x_dim / 2, y_dim / 2)

    for t in range(duration):
        x, y = path_func(t, center=center, direction=direction, space_dim=space_dim)

        x = max(min(x, x_dim - bar_size[1] - 1), 0)

        y = max(min(y, y_dim - bar_size[0] - 1), 0)

        space_time_matrix[t, x:x + bar_size[1], y:y + bar_size[0]] = 255

    return space_time_matrix


# Parameters

duration = 100  # total frames

bar_size = (1, 50)  # (height, width) of the bar

space_dim = (100, 100)  # dimension of the space

direction = (1, 0)  # L-R direction

# Create the movie

space_time_matrix = create_square(duration, bar_size, space_dim, linear_path, direction)

# Visualizing the last position of the bar in L-R direction

plt.imshow(space_time_matrix[-1, :, :], cmap='gray')

plt.title("Last Position of the Bar (L-R)")

plt.show()
plt.savefig("bar_position.png")