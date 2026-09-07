# LS test stimuli
# Sept 2025

# %%
import math
import numpy as np
import h5py
import hdf5plugin
import matplotlib.pyplot as plt

# %% A) Create a white moving bar (LHS to RHS)
stimulus = np.zeros((500, 500, 500), dtype=np.uint8) # 500 frames of 500x500 pixels each, set to 0 (black)
for i in range(499): # for all frames
    stimulus[i, :, i] = 255 # set the column i across all rows to 255 (white) - i.e. creates a moving bar

# %% B) Create a red moving bar (LHS to RHS)
stimulus_red = np.zeros((500, 500, 500, 3), dtype=np.uint8) # 500 frames of 500x500 pixels each, with 3 colour channels (RGB)
for i in range(499):
    stimulus_red[i, :, i, 0] = 255 # selecting the red channel (0) and setting to full intensity (255)

# %% B) take 2
stimulus_red = np.zeros((500, 500, 500, 4), dtype=np.uint8)

for i in range(500):
    stimulus_red[i, :, i, 0] = 255  # Red
    stimulus_red[i, :, i, 3] = 255  # Alpha (fully visible)
# %% C) Create a green moving bar (RHS to LHS)
stimulus_green = np.zeros((500, 500, 500, 3), dtype=np.uint8) # 100 frames of 500x500 pixels each, with 3 colour channels (RGB)
for i in range(499):
    stimulus_green[i, :, -i-1, 1] = 255 # selecting the red channel (0) and setting to full intensity (255)

# %% D) Create a stimulus with one moving red bar LHS-->RHS followed by moving green bar RHS-->LHS
stimulus_moving_bars = np.zeros((1000, 500, 500, 3), dtype=np.uint8)
for i in range(0,500):
    stimulus_moving_bars[i, :, i, 0] = 255
for i in range(500,1000):
    j = i - 500  # reset to 0 to 499
    stimulus_moving_bars[i, :, -(j+1), 1] = 255

# %% E) Create a stimulus with a THICKER one moving red bar LHS-->RHS followed by moving green bar RHS-->LHS
stimulus_moving_bars_thick = np.zeros((1000, 500, 500, 3), dtype=np.uint8)
for i in range(0,500):
    stimulus_moving_bars_thick[i, :, i:i+20, 0] = 255
for i in range(500,1000):
    j = i-500 # reset j to 0 to 499
    stimulus_moving_bars_thick[i, :, 500-j-20:500-j, 1] = 255

# %% F) Create a stimulus with a THICKER one moving red bar LHS-->RHS followed by moving green bar RHS-->LHS, followed by top to bottom and bottom to top
stimulus_moving_bars_thick_four = np.zeros((2000, 500, 500, 3), dtype=np.uint8)
for i in range(0,500):
    stimulus_moving_bars_thick_four[i, :, i:i+20, 0] = 255
for i in range(500,1000):
    j = i-500 # reset j to 0 to 499
    stimulus_moving_bars_thick_four[i, :, 500-j-20:500-j, 1] = 255
for i in range(1000,1500):
    j = i -1000  # reset j to 0 to 499
    stimulus_moving_bars_thick_four[i, j:j+20 ,:, 0] = 255
for i in range(1500,2000):
    j = i-1500 # reset j to 0 to 499
    stimulus_moving_bars_thick_four[i, 500-j-20:500-j, :, 1] = 255

# %% G) Create a RGB circle
stimulus_circle = np.zeros((500, 500, 500, 3), dtype=np.uint8)

centre = (250,250)
radius = 50

y, x = np.ogrid[:500, :500]
mask = (x - centre[1])**2 + (y - centre[0])**2 <= radius**2

for i in range(150,250):
    stimulus_circle[i,mask,0] = 255
for i in range(250,350):
    stimulus_circle[i,mask,1] = 255
for i in range(350,450):
    stimulus_circle[i,mask,2] = 255

# %% H) Create a blue circle that expands
stimulus_expanding_circle = np.zeros((500,500,500,3), dtype=np.uint8)

for i in range(150,450):
    radius = i-100
    y, x = np.ogrid[:500, :500]
    mask = (x - centre[1])**2 + (y - centre[0])**2 <= radius**2
    stimulus_expanding_circle[i,mask,2]=255

# %% Create stimuli file (.h5)
with h5py.File("./stimuli/LS_test_.h5", "w") as f:
    f.create_dataset(
        "Noise",
        data=stimulus_red,
        dtype="uint8",
        compression=hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
        ),
    )
    f.create_dataset(name="Frame_Rate", data=30, dtype="uint8")
    f.create_dataset(name="Checkerboard_Size", data=200, dtype="uint64")
    f.create_dataset(name="Shuffle", data=False, dtype="bool")

# %%
with h5py.File("stimuli/LS_test_.h5", "r") as f:
    print(f["Noise"].shape)  # Should be (500, 500, 500, 3)