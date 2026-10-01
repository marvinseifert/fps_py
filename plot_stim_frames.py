"""Browse the frames of an fpspy stimulus file (.h5), old or new format.

Usage:
    uv run plot_stim_frames.py /mnt/Data/boxsync/MEA/stimuli/teststim.h5

Spatial stimuli are shown as images. Full-field stimuli (1x1 pixel) are shown
as a solid colour patch, plus a trace of the value over time. A slider steps
through all frames, including repeats implied by the channel mask.
"""
import argparse
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the compression filters used by fpspy.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import fpspy


def to_image(frame: np.ndarray) -> np.ndarray:
    """Convert an (H, W, C) frame into something imshow can display.

    1 channel -> grayscale (H, W); 3 channels -> RGB; any other number of
    channels -> the channels tiled side by side as grayscale.
    """
    n_channels = frame.shape[-1]
    if n_channels == 1:
        return frame[..., 0]
    if n_channels == 3:
        return frame
    return np.concatenate([frame[..., c] for c in range(n_channels)], axis=1)


def full_field_trace(stim: fpspy.StimArray) -> np.ndarray:
    """Value of every frame of a full-field stimulus, shape (N*F, C)."""
    n_frames = stim.total_frames()
    return np.stack([stim.frame_at(i)[0, 0] for i in range(n_frames)])


def plot_trace(ax, stim: fpspy.StimArray, trace: np.ndarray):
    """Plot the per-channel value of a full-field stimulus over time."""
    times = stim.frame_times()
    for c in range(trace.shape[1]):
        ax.stairs(trace[:, c], times, label=f"channel {c}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value (display-encoded)")
    ax.legend(loc="upper right", fontsize="small")
    return ax.axvline(0, color="k", lw=1)


def frame_title(stim: fpspy.StimArray, idx: int) -> str:
    start_time = stim.frame_start_times()[idx]
    return f"frame {idx} / {stim.total_frames() - 1}   t = {start_time:.3f} s"


def show(path: Path):
    stim = fpspy.StimArray.read_hdf5(path)
    N, F, H, W, C = stim.shape
    is_full_field = H == 1 and W == 1
    print(f"{path.name}: label={stim.label!r}, shape (N, F, H, W, C) = {stim.shape}, "
          f"dtype={stim.dtype}, fps={stim.fps(allow_estimate=True)}")

    fig = plt.figure(figsize=(10, 7) if is_full_field else (7, 8))
    fig.suptitle(f"{path.name}\n{stim.label or ''}", fontsize="small")
    if is_full_field:
        ax_img = fig.add_axes([0.05, 0.45, 0.25, 0.4])
        ax_trace = fig.add_axes([0.4, 0.45, 0.55, 0.4])
        cursor = plot_trace(ax_trace, stim, full_field_trace(stim))
    else:
        ax_img = fig.add_axes([0.05, 0.15, 0.9, 0.72])
        cursor = None
    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.04])

    image = ax_img.imshow(
        to_image(stim.frame_at(0)), cmap="gray", vmin=0, vmax=255, interpolation="nearest"
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(frame_title(stim, 0), fontsize="small")

    slider = Slider(ax_slider, "frame", 0, stim.total_frames() - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(val)
        image.set_data(to_image(stim.frame_at(idx)))
        ax_img.set_title(frame_title(stim, idx), fontsize="small")
        if cursor is not None:
            cursor.set_xdata([stim.frame_start_times()[idx]] * 2)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    stim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Path to a stimulus .h5 file")
    show(parser.parse_args().path)


if __name__ == "__main__":
    main()
