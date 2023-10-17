import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import numpy as np
import hdf5plugin

path = Path(r"C:\Users\Marvin\github_packages\noisepy\stimuli")

stimulus = "shuffle_test.h5"

with h5py.File(path / stimulus, "r") as f:
    noise = np.asarray(f["Noise"])

# %%
fig, ax = plt.subplots()
ax.imshow(np.mean(noise, axis=0))
fig.show()