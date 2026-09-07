import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set file and path
file = "8px_15Hz_35mins_shuffle_x8"
h5_path = Path(r"F:\Laura\stimuli\stimuli_July_2026") / f"{file}.h5"

# Print dataset files
with h5py.File(h5_path, "r") as h5_file:
    h5_file.visit(print)

# Define dataset of interest
dataset_name = "Noise"

# Load the noise frames
with h5py.File(h5_path, "r") as h5_file:
    frames = h5_file[dataset_name][:]

# Print the noise array shape
print("Frames shape:", frames.shape)
# %% MEAN
mean_frame = np.mean(frames, axis=0) # Take mean across frames
print("Mean frame shape:", mean_frame.shape) # Print mean frame shape i.e. should be 1 frame, H, W

# Display the mean frame
plt.figure(figsize=(8, 8))
# NOTE: set interpolation to "None" as otherwise matplotlib can make the noise look weird, adding a "shadow"
plt.imshow(mean_frame, interpolation = "None", cmap="gray") # Note: default is origin="upper" for matplotlib, means 0,0 top left of plot
plt.colorbar(label="Mean pixel value")
plt.title(f"Mean of frames (n = {frames.shape[0]})\n{file}")
plt.xlabel("pixels")
plt.ylabel("pixels")
plt.show()

# %% DISPLAY ONE FRAME
frame_number = 0    # set frame of interest
plt.figure(figsize=(8, 8))
plt.imshow(frames[frame_number], interpolation= "None", cmap = "gray")
plt.colorbar(label="pixel value")
plt.title(f"Frame {frame_number}\n{file}")
plt.xlabel("pixels")
plt.ylabel("pixels")
plt.show()

