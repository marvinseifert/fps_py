"""Make extra *_15Hz_20mins_shuffle_x4 noise sets (B, C, D, E) with frames different from each other and
from the existing A file in stimuli_Nov_2026 (27 Nov 2026). Two sizes:
  4px : like A_4px_15Hz_20mins_shuffle_x4.h5  (800 x 1000 px, shifts 0/1/2/3 px)
  32px: like A_32px_15Hz_20mins_shuffle_x4.h5 (generated 800 x 1024 px as the Oct 2026 32px file, shifts
        0/8/16/24 px, then cut to the left 1000 columns -> 800 x 1000 like every stimulus in stimuli_Nov_2026;
        the Nov A file is the Oct file cut the same way)

Uses Laura's shuffle noise (shuffle_noise_LS.shuffle_pattern + create_noise.generate_checkerboard_pattern),
with shuffle_size_px = 4 (4 shift positions per axis, step = square size / 4; shuffle_noise_LS hard-codes 12,
which fails for 4 px squares). Matches file A exactly in format: 18000 frames (15 Hz x 20 min), values 0/255, dataset 'Noise' uint8 with Blosc(blosclz, clevel 9, no shuffle) and chunks (282, 25, 32),
plus Frame_Rate (uint8), Checkerboard_Size (uint64), Shuffle (bool).
Frames are written in batches of one chunk (282 frames) so the whole stack never sits in memory
(~14 GB per set otherwise, twice that while stacking).

Each set has its own seed (np.random.seed), so it can be regenerated exactly (A files had no fixed seed):
    4px : B 202611272  C 202611273  D 202611274  E 202611275
    32px: B 322611272  C 322611273  D 322611274  E 322611275

    4px no shuffle (A, B, C; <letter>_4px_15Hz_20mins_no_shuffle.h5, 800 x 1000 px, fixed grid, Shuffle = False):
          A 402611271  B 402611272  C 402611273

Run from the fps_py folder:
    python stimuli/laura_scripts/make_shuffle_noise_sets.py 4 B C              # square size, set letters
    python stimuli/laura_scripts/make_shuffle_noise_sets.py 4 --no-shuffle A B C
"""
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # fps_py (create_noise)
sys.path.insert(0, str(Path(__file__).resolve().parent))           # laura_scripts (shuffle_noise_LS)
from create_noise import generate_checkerboard_pattern  # noqa: E402
from shuffle_noise_LS import shuffle_pattern  # noqa: E402

OUT_DIR = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Nov_2026")
SIZES = {  # square size -> frame shape of that size's file A, seeds
    4: dict(height=800, width=1000, seeds={"B": 202611272, "C": 202611273, "D": 202611274, "E": 202611275}),
    32: dict(height=800, width=1024, crop_width=1000, seeds={"B": 322611272, "C": 322611273, "D": 322611274, "E": 322611275}),
}
NO_SHUFFLE_SEEDS = {4: {"A": 402611271, "B": 402611272, "C": 402611273}}
FRAMES, FPS = 18000, 15
CHUNKS = (282, 25, 32)               # chunk shape of both A files


def make_set(checker, letter, shuffle=True):
    cfg = SIZES[checker]
    HEIGHT, WIDTH, CHECKER = cfg["height"], cfg["width"], checker
    name = f"{letter}_{checker}px_15Hz_20mins_shuffle_x4.h5" if shuffle else f"{letter}_{checker}px_15Hz_20mins_no_shuffle.h5"
    out = OUT_DIR / name
    if out.exists():
        raise SystemExit(f"{out} already exists - not overwriting")
    np.random.seed(cfg["seeds"][letter] if shuffle else NO_SHUFFLE_SEEDS[checker][letter])
    with h5py.File(out, "w") as f:
        out_w = cfg.get("crop_width", WIDTH)          # 32px: generate 1024 wide, keep the left 1000 columns
        d = f.create_dataset("Noise", shape=(FRAMES, HEIGHT, out_w), dtype="uint8", chunks=CHUNKS,
                             compression=hdf5plugin.Blosc(cname="blosclz", clevel=9,
                                                          shuffle=hdf5plugin.Blosc.NOSHUFFLE))
        for start in range(0, FRAMES, CHUNKS[0]):
            n = min(CHUNKS[0], FRAMES - start)
            batch = np.empty((n, HEIGHT, WIDTH), np.uint8)
            for i in range(n):
                # create_noise returns (width_in_pixels, height_in_pixels)-shaped arrays -> pass (height, width)
                pattern = generate_checkerboard_pattern(CHECKER, HEIGHT, WIDTH)
                batch[i] = shuffle_pattern(pattern, CHECKER, shuffle_size_px=4) if shuffle else pattern
            d[start:start + n] = batch[:, :, :out_w]
            if start % (CHUNKS[0] * 16) == 0:
                print(f"{letter}: frame {start}/{FRAMES}", flush=True)
        f.create_dataset(name="Frame_Rate", data=FPS, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=CHECKER, dtype="uint64")
        f.create_dataset(name="Shuffle", data=shuffle, dtype="bool")
    print("written", out)


if __name__ == "__main__":
    checker = int(sys.argv[1])
    args = sys.argv[2:]
    shuffle = "--no-shuffle" not in args
    letters = [a for a in args if a != "--no-shuffle"] or (["B", "C", "D", "E"] if shuffle else ["A", "B", "C"])
    for letter in letters:
        make_set(checker, letter, shuffle)
