"""Merge 4px_13fps_moving_bar_ON.h5 and _OFF.h5 into one stimulus (27 Nov 2026) to repeat as a whole.

Layout of one repeat:
  ADAPT frames of black (ON background)  -> ON half (8 directions, bright bar on black)
  ADAPT frames of white (OFF background) -> OFF half (8 directions, dark bar on white)
ADAPT = 130 frames (10 s at 13 fps): the background switches black <-> white between halves and between
repeats (full-field step of maximum contrast); the pause lets the response to that step and the
adaptation settle before the first sweep.
Metadata (16 segments, ON then OFF): Directions, Direction_Start_Frame, Direction_Frame_Count,
Segment_Polarity, Segment_Background; plus Adaptation_Frames, Frame_Rate (13), Bar_* as in the sources.
Note: at 13 fps the projector (60 Hz) actually plays ~12 fps -> always time from the triggers.
"""
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

D = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Nov_2026")
SRC = [("ON", D / "4px_13fps_moving_bar_ON.h5"), ("OFF", D / "4px_13fps_moving_bar_OFF.h5")]
OUT = D / "4px_13fps_moving_bar_ON_OFF.h5"
ADAPT = 130
BLOCK = 500

if OUT.exists():
    raise SystemExit(f"{OUT} already exists - not overwriting")
srcs = [(pol, h5py.File(p, "r")) for pol, p in SRC]
H, W = srcs[0][1]["Noise"].shape[1:]
total = sum(ADAPT + f["Noise"].shape[0] for _, f in srcs)
names, starts, counts, pols, bgs = [], [], [], [], []
with h5py.File(OUT, "w") as out:
    d = out.create_dataset("Noise", shape=(total, H, W), dtype="uint8", chunks=(1, H, W),
                           compression=hdf5plugin.Blosc(cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
    pos = 0
    for pol, f in srcs:
        bg = int(f["Background_Value"][()])
        d[pos:pos + ADAPT] = np.full((ADAPT, H, W), bg, np.uint8)
        pos += ADAPT
        n = f["Noise"].shape[0]
        for s in range(0, n, BLOCK):
            d[pos + s:pos + min(s + BLOCK, n)] = f["Noise"][s:s + BLOCK]
        for nm, st, c in zip(f["Directions"][()], f["Direction_Start_Frame"][()], f["Direction_Frame_Count"][()]):
            names.append(nm); starts.append(pos + int(st)); counts.append(int(c)); pols.append(pol); bgs.append(bg)
        pos += n
        print(f"{pol}: {n} frames copied (background {bg})", flush=True)
    f0 = srcs[0][1]
    for k in ["Frame_Rate", "Checkerboard_Size", "Shuffle", "Bar_Thickness_Px", "Bar_Speed_Px_Per_Frame",
              "Ramp_Width_Px", "Stimulus_Width", "Stimulus_Height", "Pixel_Area_Antialiasing", "Antialiasing_Method"]:
        if k in f0:
            out.create_dataset(k, data=f0[k][()], dtype=f0[k].dtype)
    out.create_dataset("Directions", data=np.asarray(names, dtype="S"))
    out.create_dataset("Direction_Start_Frame", data=np.asarray(starts, dtype=np.int32))
    out.create_dataset("Direction_Frame_Count", data=np.asarray(counts, dtype=np.int32))
    out.create_dataset("Segment_Polarity", data=np.asarray(pols, dtype="S"))
    out.create_dataset("Segment_Background", data=np.asarray(bgs, dtype=np.uint8))
    out.create_dataset("Adaptation_Frames", data=ADAPT, dtype="uint16")
print("written", OUT, "| frames", total, "=", round(total / 13 / 60, 1), "min at 13 fps per repeat")
