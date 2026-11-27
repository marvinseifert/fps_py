"""Narrow moving bars for cone-scale scanning (27 Nov 2026), made with Laura's moving_bar_area_weighted.py.

Same generator, directions, pixel-area anti-aliasing and file layout as the earlier weighted moving bars;
changed settings:
  - bar core 4 px (8 µm), no ramp (0 px) -> a sharp-edged bar about one cone wide
  - 1 px per frame at 13 fps = 26 µm/s (motion blur ~2 µm over a ~70 ms response)
  - window 1000 x 800 px (width x height), the size the noise is played at, so bar and noise
    coordinates match (instead of 800 x 800)
  - ON  = bright bar (255) on a dark background (0)
  - OFF = dark bar (0) on a bright background (255)
    (the generator's OFF bar is black, so it needs a bright background to be visible)
Direction order and frame ranges are stored in each file (Directions, Direction_Start_Frame,
Direction_Frame_Count); horizontal and vertical sweeps have different lengths in a 1000 x 800 window,
so analysis must use these instead of fixed blocks.

Run from fps_py/stimuli/laura_scripts:  python make_narrow_moving_bars.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moving_bar_area_weighted as mb  # noqa: E402

OUT = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Nov_2026")
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT     # where the mp4 previews go
WIDTH, HEIGHT, FPS = 1000, 800, 13
THICK, RAMP, SPEED = 4, 0.0, 1.0

for polarity, background, label in [("ON", 0, "bright_on_dark"), ("OFF", 255, "dark_on_bright")]:
    name = f"{THICK}px_{FPS}fps_moving_bar_{polarity}"   # other details are stored inside the file
    h5 = OUT / f"{name}.h5"
    if h5.exists():
        raise SystemExit(f"{h5} already exists - not overwriting")
    mb.generate_moving_bar_stimulus(
        polarity=polarity, width=WIDTH, height=HEIGHT, fps=FPS, bar_thickness_px=THICK,
        speed_px_per_frame=SPEED, ramp_width_px=RAMP, background_value=background,
        h5_name=h5, video_name=PREVIEW / f"{name}.mp4", dataset_name="Noise", inter_direction_frames=0,
    )
