import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import numpy as np
import hdf5plugin

path = Path(r"C:\Users\Stimulus_PC\PycharmProjects\pynoise\stimuli")

stimulus = "12px_20Hz_20mins_shuffle.h5"

with h5py.File(path / stimulus, "r") as f:
    noise = np.asarray(f["Noise"])

# %%
fig, ax = plt.subplots()
ax.imshow(np.mean(noise, axis=0))
fig.savefig("noise_test.png", dpi=300)