"""moving_bar_bigger in the size of the noise window (1000 x 800 px instead of 800 x 800) (27 Nov 2026).

Same as moving_bar_bigger.h5 (fps_py/moving_bar.py): white bar (255) on black (0), 100 px (200 µm) wide,
spanning the whole window, 4 px per frame at 60 Hz (480 µm/s), binary (no anti-aliasing), same
8 directions in the same order (stimulus coordinates, row 0 = top):
  left_to_right, right_to_left, bottom_to_top, top_to_bottom,
  bottom_left_to_top_right, bottom_right_to_top_left, top_left_to_bottom_right, top_right_to_bottom_left

Difference: in a 1000 x 800 window the sweeps differ in length (horizontal 276, vertical 226, diagonal
345 frames at the same 4 px/frame), so the directions have DIFFERENT lengths: each direction = 26 blank
frames + its sweep + 50 blank frames (the margins of the original file). Use Direction_Start_Frame /
Direction_Frame_Count stored in the file to split the directions (not fixed blocks of 300).

Run from fps_py/stimuli/laura_scripts:  python make_moving_bar_bigger_noise_size.py
"""
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

OUT = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Nov_2026/moving_bar_bigger_1000x800.h5")
WIDTH, HEIGHT, FPS = 1000, 800, 60
BAR_PX, SPEED = 100, 4
PRE, POST = 26, 50                   # blank frames before / after every sweep (as in the 800 x 800 file)
S = 1 / np.sqrt(2)
DIRECTIONS = {                       # (dx, dy) in stimulus coordinates, y downwards
    "left_to_right": (1, 0), "right_to_left": (-1, 0), "bottom_to_top": (0, -1), "top_to_bottom": (0, 1),
    "bottom_left_to_top_right": (S, -S), "bottom_right_to_top_left": (-S, -S),
    "top_left_to_bottom_right": (S, S), "top_right_to_bottom_left": (-S, S),
}

yy, xx = np.mgrid[:HEIGHT, :WIDTH] + 0.5          # pixel centres
sweeps = {}
for name, (dx, dy) in DIRECTIONS.items():
    p = xx * dx + yy * dy                          # position along the direction of motion
    fronts = np.arange(p.min() - 0.5, p.max() + BAR_PX + 0.5 + SPEED, SPEED)   # bar = (front - width, front]
    sweeps[name] = (p, fronts)
counts = [PRE + len(f) + POST for _, f in sweeps.values()]
starts = list(np.cumsum([0] + counts[:-1]))

if OUT.exists():
    raise SystemExit(f"{OUT} already exists - not overwriting")
with h5py.File(OUT, "w") as f:
    d = f.create_dataset("Noise", shape=(sum(counts), HEIGHT, WIDTH), dtype="uint8",
                         compression=hdf5plugin.Blosc(cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
    for (name, (p, fronts)), start, count in zip(sweeps.items(), starts, counts):
        frames = np.zeros((count, HEIGHT, WIDTH), np.uint8)
        for i, fr in enumerate(fronts):
            frames[PRE + i][(p > fr - BAR_PX) & (p <= fr)] = 255
        d[start:start + count] = frames
        print(f"{name}: {count} frames (26 blank + sweep {len(fronts)} + 50 blank)")
    f.create_dataset("Frame_Rate", data=FPS, dtype="uint8")
    f.create_dataset("Checkerboard_Size", data=1, dtype="uint64")
    f.create_dataset("Shuffle", data=False, dtype="bool")
    f.create_dataset("Directions", data=np.asarray(list(DIRECTIONS), dtype="S"))
    f.create_dataset("Direction_Start_Frame", data=np.asarray(starts, dtype=np.int32))
    f.create_dataset("Direction_Frame_Count", data=np.asarray(counts, dtype=np.int32))
    f.create_dataset("Blank_Frames_Before_After", data=np.asarray([PRE, POST], dtype=np.int32))
    f.create_dataset("Bar_Thickness_Px", data=BAR_PX, dtype="float32")
    f.create_dataset("Bar_Speed_Px_Per_Frame", data=SPEED, dtype="float32")
    f.create_dataset("Background_Value", data=0, dtype="uint8")
    f.create_dataset("Polarity", data=np.bytes_("ON"))
print("written", OUT, "| frames", sum(counts), "=", round(sum(counts) / FPS, 1), "s per pass")
