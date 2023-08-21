import numpy as np
import h5py

duration_m = 0.5
freq = 5
canvas_size = (1000, 1000)
checkerboard_size = 1
frames = int(duration_m*60*freq)

noise_shape = (np.ceil(canvas_size[0]/checkerboard_size).astype(int),
               np.ceil(canvas_size[1]/checkerboard_size).astype(int), frames)

# %%

def generate_checkerboard_pattern(checker_size, width_in_pixels, height_in_pixels):
    # Calculate the number of squares based on pixel resolution
    pattern_width = width_in_pixels // checker_size
    pattern_height = height_in_pixels // checker_size

    pattern_shape = (pattern_height, pattern_width)
    pattern = np.random.randint(0, 2, pattern_shape, dtype=np.uint8) * 255
    pattern_texture = np.repeat(np.repeat(pattern, checker_size, axis=0), checker_size, axis=1)

    return pattern_texture







# %%
def generate_and_store_3d_array(frames, checkerboard_size, width_in_pixels, height_in_pixels):
    patterns_list = [generate_checkerboard_pattern(checkerboard_size, width_in_pixels, height_in_pixels) for _ in
                     range(frames)]
    stacked_patterns = np.stack(patterns_list, axis=0)  # This creates a 3D array
    with h5py.File('Noise.h5', 'w') as f:
        f.create_dataset('Red_Noise', data=stacked_patterns, dtype="uint8")




# Call this function once to generate and save the 3D array
generate_and_store_3d_array(frames, checkerboard_size, canvas_size[0], canvas_size[1])