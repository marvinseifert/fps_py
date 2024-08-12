import numpy as np
import h5py
import hdf5plugin

# %% Import raw stimulus

with h5py.File("./stimuli/bullseye_raw.h5", "r") as f:
    stimulus = f["Noise"][:]

frames = stimulus.shape[0]
boxes_x = stimulus.shape[1]
boxes_y = stimulus.shape[2]
boxes_x_half = boxes_x // 2
boxes_y_half = boxes_y // 2

cX = 365
cY = 400
size_x_y = 800
stacked_patterns = np.zeros((frames, size_x_y, size_x_y), dtype=np.uint8)
for i in range(frames):
    stacked_patterns[
        i, cX - boxes_x_half : cX + boxes_x_half, cY - boxes_y_half : cY + boxes_y_half
    ] = stimulus[i]


with h5py.File("./stimuli/bullseye.h5", "w") as f:
    f.create_dataset(
        "Noise",
        data=stacked_patterns,
        dtype="uint8",
        compression=hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
        ),
    )
    f.create_dataset(name="Frame_Rate", data=10, dtype="uint8")
    f.create_dataset(name="Checkerboard_Size", data=size_x_y, dtype="uint64")
    f.create_dataset(name="Shuffle", data=False, dtype="bool")
