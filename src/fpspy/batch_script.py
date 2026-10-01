"""
Order:


"""
from pathlib import Path
import typer
import fpspy.batch_3brain as batch
import fpspy._logging as _logging


app = typer.Typer()

# Just going to hard-code to our custom stim config directory.
stim_config_dir = Path(__file__).parent / "stim-config_0-0-0"
# For short:
cd = stim_config_dir

def playlist(d : Path):
    # fmt: off
    # stim to do:
        # color steps 2s
        # chirp intensity white
        # chirp frequency white
        # moving bars 8 dir white
        # moving dot horizontal (covering whole screen bidirectional)  white
        # moving dot vertical (same as above)  white
        # noise  white
    res = [
        batch.Item(
            d / "objbg/objbg_2-9-7-0-0.py",
            loops=1,
        ),
        batch.Item(
            d / "paramecia/paramecia_2-9-6-0-0.py",
            loops=1,
        ),
        batch.Item(
            d / "moving-bars-glsl/movingbar_2-0-5-0-0.py",
            #stim_config_path=d / "moving-bars-glsl/all-128tk-8s.json",
            stim_config_path=cd / "movingbar_gbw-128tk-8s.json",
            loops=3,
        ),
        batch.Item(
            d / "steps/steps-rgbw_1s_2_0_1_3_0.h5",
            loops=5,
        ),
        batch.Item(
            d / "steps/steps-rgbw_2s_2_0_1_3_0.h5",
            loops=5,
        ),
        batch.Item(
            d / "chirp/chirp_white_30Hz_2_0_0_5_0.h5",
            loops=1,
        ),
        batch.Item(
            d / "chirp/chirp_led2_2_0_0_5_0.h5",
            loops=1,
        ),
        batch.Item(
            d / "chirp/chirp_led3_2_0_0_5_0.h5",
            loops=1,
        ),
        batch.Item(
            d / "movie/linearmovie_2-9-3-2-0.py",
            stim_config_path=cd / "vid-02.json",
            loops=10,
        ),
        batch.Item(
           d / "movie/linearmovie_2-9-3-2-0.py",
           loops=5,
        ),
        batch.Item(
            d / "bluebar/bluebar_2-9-4-0-0.py",
            stim_config_path=cd / "bluebar_0-0-1.json",
            loops=5,
        ),
        batch.Item(
            d / "bluebar/bluebar_2-9-4-0-0.py",
            stim_config_path=cd / "greenbar_0-0-1.json",
            loops=5,
        ),
        batch.Item(
            d / "bounce/bounce_2-9-2-0-0.py",
            loops=3,
        ),
        batch.Item(
            d / "shadowbar/shadowbar_2-9-1-0-0.py",
            loops=3,
        ),
        batch.Item(
            d / "dotgrid/dotgrid_2-9-0-0-0.py",
            stim_config_path=cd / "dotgrid-w.json",
            loops=10,
        ),
        batch.Item(
            d / "checkerboard-noise/gaussian_checkerboard_768l-8s-20Hz-40min-p1_2_0_2_6_0.h5",
            loops=1,
            lazy_textures=True,
        ),
        batch.Item(
            d / "checkerboard-noise/gaussian_checkerboard_768l-8s-20Hz-40min-p0_2_0_2_6_0.h5",
            loops=1,
            lazy_textures=True,
        ),
        batch.Item(
            d / "checkerboard-noise-glsl/gaussian_checkerboard_2-0-6-0-0.py",
            stim_config_path=d / "checkerboard-noise-glsl/2s-20Hz-80min.json",
        ),
    ]
    # fmt: on
    return res


@app.command()
def run(stim_root : Path, dry_run : bool = False):
    _logging.setup_main_logging(log_level="INFO")

    delay = 10.0
    print(batch.info(playlist(stim_root), delay))

    if dry_run:
        return

    batch.play_playlist(
        playlist(stim_root),
        config_path="/mnt/Data/Data/fpspy-resources/fpspy-single.toml",
        enable_triggers=True,
        delay=delay,
        label="collect_2-4-11-0",
    )


if __name__ == "__main__":
    app()
