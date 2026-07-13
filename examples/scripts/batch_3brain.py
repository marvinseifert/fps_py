import fpspy.batch_3brain as batch
import fpspy._logging as _logging


def main():
    _logging.setup_main_logging(log_level="INFO")

    batch.play_playlist(
        [
            batch.Item(
                "examples/shader_based_stimuli/movingbar/movingbar.py",
                stim_config_path=(
                    "examples/shader_based_stimuli/movingbar/3led_10tk_2s.json"
                ),
            ),
            batch.Item(
                "../data/stim/gaussian_checkerboard_768l-16s-10Hz-5min.h5",
                lazy_textures=True,
            ),
        ],
        config_path="../_configs/fpspy.toml",
        enable_triggers=False,
    )


if __name__ == "__main__":
    main()
