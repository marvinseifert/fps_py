import numpy as np
import h5py


def generate_moving_box(box_width, box_height, frame_num, total_frames, width_in_pixels, height_in_pixels):
    """
    Generate an image with a moving vertical box which goes out of bounds.

    Parameters
    ----------
    box_width : int
        The width of the moving box in pixels.
    box_height : int
        The height of the moving box in pixels.
    frame_num : int
        Current frame number.
    total_frames : int
        Total number of frames.
    width_in_pixels : int
        The width of the image in pixels.
    height_in_pixels : int
        The height of the image in pixels.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the image with the moving box.

    """
    # Create an empty image
    image = np.zeros((height_in_pixels, width_in_pixels), dtype=np.uint8)

    # Determine the box's x position based on the current frame number
    start_x = frame_num*4
    end_x = start_x + box_width

    # Allow box to go out of bounds, but ensure at least 1 pixel remains visible in the last frame
    # if end_x > width_in_pixels + box_width - 1:
    #     end_x = width_in_pixels
    #     start_x = end_x - 1

    # Determine the box's y position (centered vertically)
    start_y = (height_in_pixels - box_height) // 2
    end_y = start_y + box_height

    # Fill the box region with 255 (or any other foreground value)
    image[start_y:end_y, start_x:end_x] = 255

    return image



def generate_and_store_moving_box_array(frames, box_width, box_height, width_in_pixels, height_in_pixels, fps,
                                        name="MovingBox.h5"):
    """Generate a 3D array of moving box patterns and store it in an HDF5 file.

    Parameters
    ----------
    frames : int
        The number of frames to generate.
    box_width : int
        The width of the moving box in pixels.
    box_height : int
        The height of the moving box in pixels.
    width_in_pixels : int
        The width of the pattern in pixels.
    height_in_pixels : int
        The height of the pattern in pixels.
    fps : int
        The frame rate of the pattern in Hz.
    name : str
        The name of the HDF5 file to store the pattern in.

    """

    patterns_list = [generate_moving_box(box_width, box_height, i, frames, width_in_pixels, height_in_pixels) for i in
                     range(frames)]
    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array

    with h5py.File(name, 'w') as f:
        f.create_dataset('Noise', data=stacked_patterns, dtype="uint8")
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=box_width, dtype="uint64")
        f.create_dataset(name="Shuffle", data=False, dtype="bool")

# As we also use the h5py library, we'll need to skip the actual file-writing step here
# since our environment doesn't support h5py directly.
# Instead, you can use the provided function in your local environment after installing h5py.


generate_and_store_moving_box_array(800, 50, 800, 800, 800, 30, "stimuli/MovingLine.h5")
