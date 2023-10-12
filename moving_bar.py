import numpy as np
from scipy.ndimage import binary_dilation
import h5py
import hdf5plugin
import blosc
from scipy.ndimage import rotate


DIRECTIONS = {
    "right": (2, 0),
    "left": (-2, 0),
    "up": (0, -2),
    "down": (0, 2),
    "up-right": (2, -2, 45),
    "up-left": (-2, -2, -45),
    "down-right": (2, 2, -45),
    "down-left": (-2, 2, 45)
}


def generate_stimulus(nt, bar_width):
    """
    Generate the full stimulus with all directions and repetitions.

    Parameters:
    - nt: int, number of frames per stimulus
    - bar_width: int, width of the bar

    Returns:
    - stimulus: np.array, array of shape (nt*8*10, 600, 600) containing the stimulus frames
    """
    # Total number of stimuli (8 directions x 10 repetitions)
    repeats = 1
    num_stimuli = 8 * repeats

    # Initialize an array to store the stimulus
    stimulus = np.zeros((nt * num_stimuli, 600, 600), dtype=np.uint8)

    # Generate the stimulus for each direction and repetition
    for i, direction in enumerate(DIRECTIONS.keys()):
        for rep in range(repeats):
            # Generate the moving bar stimulus for the current direction
            bar_frames = move_bar_optimized(nt, bar_width, direction)

            # Determine the index in the stimulus array to place the frames
            idx = i * repeats * nt + rep * nt

            # Store the frames in the stimulus array
            stimulus[idx:idx + nt] = bar_frames

    return stimulus


def create_large_diagonal_bar(frame_size, width, orientation):
    """

    Create a large diagonal bar that can cover the entire frame when shifted.



    Parameters:

    - frame_size: tuple, (height, width) of the frame

    - width: int, width of the bar

    - orientation: str, one of ["diagonal_right", "diagonal_left"]



    Returns:

    - large_bar: np.array, a large frame with a diagonal bar

    """

    # Creating a large frame to ensure the bar can traverse the entire frame diagonally

    large_frame = np.zeros((frame_size[0] + width, frame_size[1] + width), dtype=np.uint8)

    # Creating a thin diagonal line on the large frame

    rr, cc = np.diag_indices_from(large_frame)

    if orientation == "diagonal_left":
        cc = large_frame.shape[1] - 1 - cc

    large_frame[rr, cc] = 1

    # Dilating the thin line to create a wide bar

    large_bar = binary_dilation(large_frame, structure=np.ones((width, width)))

    return large_bar


# def move_bar_optimized(nt, width, direction, speed=2):
#     """
#     Move the bar across frames in the specified direction (optimized version).
#
#     Parameters:
#     - nt: int, number of frames
#     - width: int, width of the bar
#     - direction: str, one of the keys from DIRECTIONS dict
#     - speed: int, pixels per frame the bar moves
#
#     Returns:
#     - frames: np.array, array of shape (nt, 600, 600) with the moving bar
#     """
#     dx, dy = DIRECTIONS[direction]
#     extended_frame_size = (600 + 2 * width, 600 + 2 * width)
#     central_slice = (slice(width, width + 600), slice(width, width + 600))
#
#     frames = np.zeros((nt, 600, 600), dtype=np.uint8)
#
#     for t in range(nt):
#         # Creating an extended blank frame
#         frame_ext = np.zeros(extended_frame_size, dtype=np.uint8)
#
#         # Position for the bar in the extended frame
#         pos_x, pos_y = t * dx, t * dy
#
#         # Placing the bar on the extended frame
#         if "up" in direction or "down" in direction:
#             frame_ext[pos_y:pos_y + width, :] = 1
#         elif "left" in direction:
#             frame_ext[:, pos_x:pos_x + width] = 1
#         elif "right" in direction:
#             frame_ext[:, pos_x-width:pos_x] = 1
#         elif "up-right" in direction:
#             rr, cc = np.indices(extended_frame_size)
#             mask = np.abs(rr + cc - pos_y - pos_x) < width // np.sqrt(2)
#             frame_ext[mask] = 1
#         elif "up-left" in direction:
#             rr, cc = np.indices(extended_frame_size)
#             mask = np.abs(rr - cc - pos_y + pos_x) < width // np.sqrt(2)
#             frame_ext[mask] = 1
#         elif "down-right" in direction:
#             rr, cc = np.indices(extended_frame_size)
#             mask = np.abs(rr - cc + pos_y - pos_x) < width // np.sqrt(2)
#             frame_ext[mask] = 1
#         elif "down-left" in direction:
#             rr, cc = np.indices(extended_frame_size)
#             mask = np.abs(rr + cc + pos_y + pos_x) < width // np.sqrt(2)
#             frame_ext[mask] = 1
#
#         # Extracting the central region to create the visible frame
#         frames[t] = frame_ext[central_slice]
#
#     return frames


def move_bar_optimized(nt, width, direction, speed=2):
    dx, dy = DIRECTIONS[direction][:2]
    extended_frame_size = (600 + 2 * width, 600 + 2 * width)
    central_slice = (slice(width, width + 600), slice(width, width + 600))

    frames = np.zeros((nt, 600, 600), dtype=np.uint8)

    for t in range(nt):
        frame_ext = np.zeros(extended_frame_size, dtype=np.uint8)
        pos_x, pos_y = t * dx, t * dy

        if direction in ["up-left", "down-left"]:
            # Create a vertical bar
            frame_ext[:, pos_x-width:pos_x] = 1
            # Rotate the frame
            angle = DIRECTIONS[direction][2]
            frame_ext = rotate(frame_ext, angle, reshape=False, order=0, mode='constant', cval=0)
        elif direction in ["up-right", "down-right"]:
            # Create a vertical bar
            frame_ext[:, pos_x:pos_x + width] = 1
            # Rotate the frame
            angle = DIRECTIONS[direction][2]
            frame_ext = rotate(frame_ext, angle, reshape=False, order=0, mode='constant', cval=0)
        elif "up" in direction or "down" in direction:
            frame_ext[pos_y:pos_y + width, :] = 1
        elif "left" in direction:
            frame_ext[:, pos_x:pos_x + width] = 1
        elif "right" in direction:
            frame_ext[:, pos_x - width:pos_x] = 1

        frames[t] = frame_ext[central_slice]

    return frames

nt = 400  # number of frames per stimulus
bar_width = 100  # width of the bar

# %%
# Generate the stimulus
stimulus = generate_stimulus(nt, bar_width)

# %%
stimulus[stimulus==1] = 255
# %%
with h5py.File("stimuli/moving_bar.h5", 'w') as f:
    f.create_dataset('Noise', data=stimulus, dtype="uint8",
                     compression=hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
    f.create_dataset(name="Frame_Rate", data=60, dtype="uint8")
    f.create_dataset(name="Checkerboard_Size", data=1, dtype="uint64")
    f.create_dataset(name="Shuffle", data=False, dtype="bool")


# %%
import cv2
def save_as_video(frames, filename='output.avi', fps=30):
    """
    Save the frames as a video using OpenCV.

    Parameters:
    - frames: np.array, input frames to be saved as video, shape (num_frames, height, width)
    - filename: str, name of the output video file
    - fps: int, frames per second for the output video

    """
    num_frames, height, width = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    for i in range(num_frames):
        # OpenCV expects uint8 type for images. Scaling and converting the dtype
        frame = frames[i]
        # Adding an additional channel dimension to make it (height, width, num_channels)
        frame = np.expand_dims(frame, axis=-1)
        out.write(frame)

    out.release()


save_as_video(stimulus, filename="stimuli/moving_bar.avi", fps=60)