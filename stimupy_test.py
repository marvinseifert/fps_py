import stimupy
import matplotlib.pyplot as plt
from stimupy.utils import plot_stim
import numpy as np
from numpy.random import default_rng
import h5py
import hdf5plugin
import matplotlib
import matplotlib.animation
from stimupy.components.waves import bessel
from stimupy.stimuli.waves import square_radial
from stimupy.stimuli.cornsweets import cornsweet
from stimupy.components import shapes
import math

# %%
stim_bull = shapes.rectangle(
    visual_size=(80, 80), ppd=10, rectangle_size=(4, 2), rectangle_position=(1, 3)
)
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(np.roll(stim_bull["img"], 1, axis=1), cmap="Greys")
fig.show()

# %%
stacked_patterns = np.zeros((10, 800, 800), dtype=np.uint8)
for frame in range(10):
    stacked_patterns[frame, :, :] = np.roll(stim_bull["img"], frame, axis=1)

stacked_patterns = stacked_patterns * 255


# %%

with h5py.File("./stimuli/moving_bar_test.h5", "w") as f:
    f.create_dataset(
        "Noise",
        data=stacked_patterns,
        dtype="uint8",
        compression=hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
        ),
    )
    f.create_dataset(name="Frame_Rate", data=1, dtype="uint8")
    f.create_dataset(name="Checkerboard_Size", data=200, dtype="uint64")
    f.create_dataset(name="Shuffle", data=False, dtype="bool")


# %%
frames = 3000
stacked_patterns = np.zeros((frames, 800, 800), dtype=np.uint8)

radii_array = np.zeros((frames, 5), dtype=float)
for i in range(frames):
    # Create 5 random floats between 0 and 15
    while True:
        rng = default_rng()
        radii = np.sort(rng.uniform(0.01, 20, 6))
        radii = radii[1:][np.diff(radii) > 0.1]
        if len(radii) > 1:
            radii_array[i, : len(radii)] = radii
            break
radii_array = np.sort(radii_array, axis=1)

# %%
# np.save("radii_array.npy", radii_array)
# %%
stimulus = np.zeros((frames, 400, 400), dtype=float)
for i, rad_frame in enumerate(radii_array):
    if np.random.choice([True, False]):
        stim_gen = stimupy.bullseyes.circular_generalized(
            visual_size=(40, 40),
            ppd=10,
            radii=rad_frame,
            intensity_target=0,
            intensity_background=0.5,
            intensity_rings=(1, 0),
        )
    else:
        stim_gen = stimupy.bullseyes.circular_generalized(
            visual_size=(40, 40),
            ppd=10,
            radii=rad_frame,
            intensity_target=1,
            intensity_background=0.5,
            intensity_rings=(0, 1),
        )

    stimulus[i, :, :] = stim_gen["img"]

# %%
np.save("bullseye_raw.npy", stimulus)

# %%
pattern_mean = np.mean(stimulus, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
im = ax.imshow(pattern_mean, cmap="Greys", vmin=0, vmax=1)
plt.colorbar(im)
fig.show()


# %%
stimulus = stimulus * 255


# %%
cX = 200
cY = 400
stacked_patterns = np.zeros((frames, 800, 800), dtype=np.uint8)
for i in range(frames):
    stacked_patterns[i, cX - 200 : cX + 200, cY - 200 : cY + 200] = stimulus[i]
# %%


def kernel_mov(ker):
    n_frames, h, w = ker.shape
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(ker[0], cmap="Greys_r")

    def update(frame):
        im.set_array(ker[frame])
        return [im]

    animation = matplotlib.animation.FuncAnimation(
        fig, update, frames=range(n_frames), blit=True
    )

    return animation


# %%
ani = kernel_mov(stacked_patterns)
ani.save("moving_bar_test.mp4", writer="ffmpeg", fps=10)
# %%

with h5py.File("./stimuli/ring_noise.h5", "w") as f:
    f.create_dataset(
        "Noise",
        data=stacked_patterns,
        dtype="uint8",
        compression=hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE
        ),
    )
    f.create_dataset(name="Frame_Rate", data=10, dtype="uint8")
    f.create_dataset(name="Checkerboard_Size", data=200, dtype="uint64")
    f.create_dataset(name="Shuffle", data=False, dtype="bool")


# %%
frames = 3000
stacked_patterns = np.zeros((frames, 600, 600), dtype=float)


# %%


for i in range(frames):
    # Create a random frequency float
    radius = np.random.uniform(0.01, 2)
    disc = shapes.disc(
        visual_size=(5, 5),
        ppd=10,
        radius=radius,
        intensity_disc=1,
        intensity_background=0,
    )
    nr = int(600 * 600 / (radius * 10 * math.pi) * 0.5 / 10)
    # if np.random.choice([True, False]):
    #     stim_gen["img"] = np.logical_not(stim_gen["img"]).astype(np.uint8)
    for pos in range(nr):
        pos_x = np.random.randint(50, 550)
        pos_y = np.random.randint(50, 550)
        stacked_patterns[
            i,
            pos_x - 25 : pos_x + 25,
            pos_y - 25 : pos_y + 25,
        ] = np.logical_or(
            stacked_patterns[
                i,
                pos_x - 25 : pos_x + 25,
                pos_y - 25 : pos_y + 25,
            ],
            disc["img"],
        )

stacked_patterns = (stacked_patterns * 255).astype(np.uint8)

# %%
ani = kernel_mov(stacked_patterns)
ani.save("ring_noise.mp4", writer="ffmpeg", fps=1)

# %%
pattern_mean = np.mean(stacked_patterns, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
im = ax.imshow(pattern_mean, cmap="Greys")
plt.colorbar(im)
fig.show()
