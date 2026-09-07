# Alignment pipeline sanity check noise

# %%
# Uniform grey 800px x 800px noise, with top left quadrant only containing noise (this is top left because the default of matplotlib is origin upper)
# Noise is 40px checkers, not shuffled, 15Hz

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def generate_checkerboard_pattern(checker_size, width_in_pixels, height_in_pixels):
    """
    Generate a random black/white checkerboard pattern and crop it
    to the requested size.
    """
    n_rows = int(np.ceil(height_in_pixels / checker_size))
    n_cols = int(np.ceil(width_in_pixels / checker_size))

    # Random binary board: 0 or 255
    small_board = np.random.randint(0, 2, size=(n_rows, n_cols), dtype=np.uint8) * 255

    # Expand each checker
    pattern = np.repeat(np.repeat(small_board, checker_size, axis=0), checker_size, axis=1)

    # Crop to exact size
    return pattern[:height_in_pixels, :width_in_pixels].astype(np.uint8)


def generate_and_store_3d_array_top_left_noise(
    width_in_pixels: int = 800,
    height_in_pixels: int = 800,
    fps: int = 15,
    checkerboard_size: int = 32,
    duration_minutes: int = 20,
    background_value: int = 127,
    name: str | Path = "Noise.h5",
):
    """
    Generate a 3D array of frames where only the top-right quadrant
    contains checkerboard noise and the other three quadrants are
    uniform grey, then store it in an HDF5 file.

    Layout:
        top-left     = grey
        top-right    = noise
        bottom-left  = grey
        bottom-right = grey
    """

    total_frames = duration_minutes * 60 * fps

    quadrant_height = height_in_pixels // 2
    quadrant_width = width_in_pixels // 2

    patterns_list = []

    for _ in range(total_frames):
        # Start with a uniform grey frame
        frame = np.full(
            (height_in_pixels, width_in_pixels),
            fill_value=background_value,
            dtype=np.uint8,
        )

        # Generate noise for top-right quadrant only
        noise_quadrant = generate_checkerboard_pattern(
            checker_size=checkerboard_size,
            width_in_pixels=quadrant_width,
            height_in_pixels=quadrant_height,
        )

        # Insert into top-left quadrant
        frame[0:quadrant_height, 0:quadrant_width] = noise_quadrant
        #frame[0:quadrant_height, quadrant_width:width_in_pixels] = noise_quadrant # top right quadrant

        patterns_list.append(frame)

    stacked_patterns = np.stack(patterns_list, axis=0)

    with h5py.File(name, "w") as f:
        f.create_dataset(
            "Noise",
            data=stacked_patterns,
            dtype="uint8",
            compression=hdf5plugin.Blosc(
                cname="blosclz",
                clevel=9,
                shuffle=hdf5plugin.Blosc.NOSHUFFLE,
            ),
        )
        f.create_dataset(name="Frame_Rate", data=fps, dtype="uint8")
        f.create_dataset(name="Checkerboard_Size", data=checkerboard_size, dtype="uint64")
        f.create_dataset(name="Duration_Minutes", data=duration_minutes, dtype="uint64")
        f.create_dataset(name="Shuffle", data=False, dtype="bool")


if __name__ == "__main__":
    generate_and_store_3d_array_top_left_noise(
        width_in_pixels=800,
        height_in_pixels=800,
        fps=15,
        checkerboard_size=32,
        duration_minutes=20,
        background_value=127,
        name="32px_15Hz_20mins_top_left_noise.h5",
    )

# %%
# Alignment pipeline asymmetric sanity-check noise
#
# Matplotlib / NumPy orientation: origin="upper", so [0, 0] is top-left.
#
# Active regions:
#   1. Entire top-left quadrant
#   2. Upper half of the top-right quadrant
#
# All other pixels remain uniform grey.
#
# Stimulus:
#   800 × 800 px
#   32 px checkers
#   15 Hz
#   20 minutes

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def generate_checkerboard_pattern(
    checker_size: int,
    width_in_pixels: int,
    height_in_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random black-and-white checkerboard pattern.

    Parameters
    ----------
    checker_size
        Width and height of each checker in pixels.
    width_in_pixels
        Required output width.
    height_in_pixels
        Required output height.
    rng
        NumPy random-number generator.

    Returns
    -------
    np.ndarray
        Array with shape (height_in_pixels, width_in_pixels),
        containing values 0 and 255.
    """

    n_rows = int(np.ceil(height_in_pixels / checker_size))
    n_cols = int(np.ceil(width_in_pixels / checker_size))

    # Random checker values: black=0, white=255
    small_board = rng.integers(
        low=0,
        high=2,
        size=(n_rows, n_cols),
        dtype=np.uint8,
    ) * 255

    # Expand each value into a checker_size × checker_size square
    pattern = np.repeat(
        np.repeat(
            small_board,
            checker_size,
            axis=0,
        ),
        checker_size,
        axis=1,
    )

    # Crop to the exact requested dimensions
    return pattern[
        :height_in_pixels,
        :width_in_pixels,
    ].astype(np.uint8)


def generate_and_store_asymmetric_noise(
    width_in_pixels: int = 800,
    height_in_pixels: int = 800,
    fps: int = 15,
    checkerboard_size: int = 32,
    duration_minutes: int = 20,
    background_value: int = 127,
    random_seed: int | None = 42,
    name: str | Path = "32px_15Hz_20mins_asymmetric_noise.h5",
) -> None:
    """
    Generate an asymmetric checkerboard-noise stimulus.

    Matplotlib/NumPy layout with origin="upper":

        ┌─────────────────┬─────────────────┐
        │                 │                 │
        │   NOISE         │   NOISE         │
        │                 │                 │
        ├                 ├─────────────────┤
        │                 │                 │
        │   NOISE         │   GREY          │
        │                 │                 │
        ├─────────────────┼─────────────────┤
        │                                   │
        │               GREY                │
        │                                   │
        └───────────────────────────────────┘

    In other words:

        - full top-left quadrant = noise
        - upper half of top-right quadrant = noise
        - remainder = grey
    """

    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration_minutes * 60 * fps)

    quadrant_height = height_in_pixels // 2
    quadrant_width = width_in_pixels // 2

    # The top-right active region is only half the height of a quadrant
    top_right_half_height = quadrant_height // 2

    rng = np.random.default_rng(random_seed)

    print(f"Output: {name}")
    print(f"Frames: {total_frames:,}")
    print(
        f"Stimulus dimensions: "
        f"{height_in_pixels} × {width_in_pixels} px"
    )
    print(f"Checker size: {checkerboard_size} px")
    print(f"Frame rate: {fps} Hz")

    with h5py.File(name, "w") as h5_file:

        noise_dataset = h5_file.create_dataset(
            name="Noise",
            shape=(
                total_frames,
                height_in_pixels,
                width_in_pixels,
            ),
            dtype=np.uint8,
            chunks=(
                1,
                height_in_pixels,
                width_in_pixels,
            ),
            compression=hdf5plugin.Blosc(
                cname="blosclz",
                clevel=9,
                shuffle=hdf5plugin.Blosc.NOSHUFFLE,
            ),
        )

        for frame_index in range(total_frames):

            # Begin with a completely grey frame
            frame = np.full(
                shape=(height_in_pixels, width_in_pixels),
                fill_value=background_value,
                dtype=np.uint8,
            )

            # Generate noise across the full top half.
            #
            # We then copy only the required parts into the frame:
            #   - all of its left half
            #   - only the upper half of its right half
            top_half_noise = generate_checkerboard_pattern(
                checker_size=checkerboard_size,
                width_in_pixels=width_in_pixels,
                height_in_pixels=quadrant_height,
                rng=rng,
            )

            # --------------------------------------------------
            # Entire top-left quadrant
            # --------------------------------------------------
            frame[
                0:quadrant_height,
                0:quadrant_width,
            ] = top_half_noise[
                0:quadrant_height,
                0:quadrant_width,
            ]

            # --------------------------------------------------
            # Upper half of top-right quadrant
            # --------------------------------------------------
            frame[
                0:top_right_half_height,
                quadrant_width:width_in_pixels,
            ] = top_half_noise[
                0:top_right_half_height,
                quadrant_width:width_in_pixels,
            ]

            # Write this frame directly to disk
            noise_dataset[frame_index] = frame

            if (
                frame_index % 1000 == 0
                or frame_index == total_frames - 1
            ):
                print(
                    f"Written frame "
                    f"{frame_index + 1:,}/{total_frames:,}"
                )

        # Metadata expected by the presentation software
        h5_file.create_dataset(
            name="Frame_Rate",
            data=fps,
            dtype="uint8",
        )

        h5_file.create_dataset(
            name="Checkerboard_Size",
            data=checkerboard_size,
            dtype="uint64",
        )

        h5_file.create_dataset(
            name="Duration_Minutes",
            data=duration_minutes,
            dtype="uint64",
        )

        h5_file.create_dataset(
            name="Shuffle",
            data=False,
            dtype="bool",
        )

    print("Finished saving stimulus.")


if __name__ == "__main__":
    generate_and_store_asymmetric_noise(
        width_in_pixels=800,
        height_in_pixels=800,
        fps=15,
        checkerboard_size=32,
        duration_minutes=20,
        background_value=127,
        random_seed=42,
        name=Path(
            "stimuli/laura_scripts"
            "32px_15Hz_20mins_top_left_L_shape.h5"
        ),
    )

# %%
# %%
# Shuffled asymmetric L-shaped checkerboard noise
#
# Matplotlib / NumPy orientation: origin="upper", so [0, 0] is top-left.
#
# Active regions before shuffle:
#   1. Entire top-left quadrant
#   2. Upper half of the top-right quadrant
#
# All other pixels remain uniform grey.
#
# Then the ENTIRE 800 x 800 frame is shuffled using np.roll(),
# so the whole pattern is shifted by one of 4 checker phases in x and y.
#
# Stimulus:
#   800 x 800 px
#   32 px checkers
#   15 Hz
#   20 minutes

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def generate_checkerboard_pattern(
    checker_size: int,
    width_in_pixels: int,
    height_in_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random black-and-white checkerboard pattern.

    Parameters
    ----------
    checker_size
        Width and height of each checker in pixels.
    width_in_pixels
        Output width in pixels.
    height_in_pixels
        Output height in pixels.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (height_in_pixels, width_in_pixels)
        containing values 0 and 255.
    """

    n_rows = int(np.ceil(height_in_pixels / checker_size))
    n_cols = int(np.ceil(width_in_pixels / checker_size))

    small_board = rng.integers(
        low=0,
        high=2,
        size=(n_rows, n_cols),
        dtype=np.uint8,
    ) * 255

    pattern = np.repeat(
        np.repeat(small_board, checker_size, axis=0),
        checker_size,
        axis=1,
    )

    return pattern[:height_in_pixels, :width_in_pixels].astype(np.uint8)


def shuffle_pattern(
    pattern: np.ndarray,
    checker_size: int,
    rng: np.random.Generator,
    shuffle_positions: int = 4,
) -> np.ndarray:
    """
    Shuffle the ENTIRE frame by discrete phase shifts in x and y.

    For example, with checker_size = 32 and shuffle_positions = 4,
    the allowed shifts are:
        [0, 8, 16, 24]

    Parameters
    ----------
    pattern
        2D image array to shift.
    checker_size
        Checker width/height in pixels.
    rng
        NumPy random number generator.
    shuffle_positions
        Number of equally spaced phase positions within one checker.

    Returns
    -------
    np.ndarray
        Shifted version of the input array.
    """

    if shuffle_positions < 1:
        raise ValueError("shuffle_positions must be >= 1")

    shift_step = checker_size // shuffle_positions
    if shift_step < 1:
        raise ValueError(
            "checker_size is too small relative to shuffle_positions"
        )

    shifts = np.arange(0, checker_size, shift_step, dtype=int)

    x_shift = int(rng.choice(shifts))
    y_shift = int(rng.choice(shifts))

    shifted_pattern = np.roll(pattern, shift=x_shift, axis=1)  # x
    shifted_pattern = np.roll(shifted_pattern, shift=y_shift, axis=0)  # y

    return shifted_pattern


def generate_and_store_shuffled_l_shaped_noise(
    width_in_pixels: int = 800,
    height_in_pixels: int = 800,
    fps: int = 15,
    checkerboard_size: int = 32,
    duration_minutes: int = 20,
    background_value: int = 127,
    random_seed: int | None = 42,
    shuffle_positions: int = 4,
    name: str | Path = "32px_15Hz_20mins_top_left_L_shape_shuffle.h5",
) -> None:
    """
    Generate an asymmetric L-shaped checkerboard-noise stimulus and
    shuffle the ENTIRE frame on every time step.

    Base unshuffled layout:

        ┌─────────────────┬─────────────────┐
        │                 │                 │
        │   NOISE         │   NOISE         │
        │                 │                 │
        ├                 ├─────────────────┤
        │                 │                 │
        │   NOISE         │   GREY          │
        │                 │                 │
        ├─────────────────┼─────────────────┤
        │                                   │
        │               GREY                │
        │                                   │
        └───────────────────────────────────┘

    i.e.
        - full top-left quadrant = noise
        - upper half of top-right quadrant = noise
        - remainder = grey

    The whole 2D frame is then shuffled with np.roll().
    """

    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration_minutes * 60 * fps)

    quadrant_height = height_in_pixels // 2
    quadrant_width = width_in_pixels // 2
    top_right_half_height = quadrant_height // 2

    rng = np.random.default_rng(random_seed)

    print(f"Output file: {name}")
    print(f"Total frames: {total_frames:,}")
    print(f"Stimulus size: {height_in_pixels} x {width_in_pixels} px")
    print(f"Checker size: {checkerboard_size} px")
    print(f"Frame rate: {fps} Hz")
    print(f"Duration: {duration_minutes} min")
    print(f"Shuffle positions per axis: {shuffle_positions}")

    shift_step = checkerboard_size // shuffle_positions
    shifts = np.arange(0, checkerboard_size, shift_step, dtype=int)
    print(f"Possible shifts (px): {shifts.tolist()}")

    with h5py.File(name, "w") as h5_file:
        noise_dataset = h5_file.create_dataset(
            name="Noise",
            shape=(total_frames, height_in_pixels, width_in_pixels),
            dtype=np.uint8,
            chunks=(1, height_in_pixels, width_in_pixels),
            compression=hdf5plugin.Blosc(
                cname="blosclz",
                clevel=9,
                shuffle=hdf5plugin.Blosc.NOSHUFFLE,
            ),
        )

        for frame_index in range(total_frames):
            # --------------------------------------------------
            # 1. Start with a grey full-frame image
            # --------------------------------------------------
            frame = np.full(
                shape=(height_in_pixels, width_in_pixels),
                fill_value=background_value,
                dtype=np.uint8,
            )

            # --------------------------------------------------
            # 2. Generate random noise over the top half
            # --------------------------------------------------
            top_half_noise = generate_checkerboard_pattern(
                checker_size=checkerboard_size,
                width_in_pixels=width_in_pixels,
                height_in_pixels=quadrant_height,
                rng=rng,
            )

            # --------------------------------------------------
            # 3. Insert full top-left quadrant
            # --------------------------------------------------
            frame[
                0:quadrant_height,
                0:quadrant_width,
            ] = top_half_noise[
                0:quadrant_height,
                0:quadrant_width,
            ]

            # --------------------------------------------------
            # 4. Insert upper half of top-right quadrant
            # --------------------------------------------------
            frame[
                0:top_right_half_height,
                quadrant_width:width_in_pixels,
            ] = top_half_noise[
                0:top_right_half_height,
                quadrant_width:width_in_pixels,
            ]

            # --------------------------------------------------
            # 5. Shuffle the ENTIRE frame
            # --------------------------------------------------
            frame = shuffle_pattern(
                pattern=frame,
                checker_size=checkerboard_size,
                rng=rng,
                shuffle_positions=shuffle_positions,
            )

            # --------------------------------------------------
            # 6. Write frame directly to disk
            # --------------------------------------------------
            noise_dataset[frame_index] = frame

            if (
                frame_index % 1000 == 0
                or frame_index == total_frames - 1
            ):
                print(
                    f"Written frame {frame_index + 1:,}/{total_frames:,}"
                )

        # ------------------------------------------------------
        # Metadata expected by presentation software
        # ------------------------------------------------------
        h5_file.create_dataset(
            name="Frame_Rate",
            data=fps,
            dtype="uint8",
        )

        h5_file.create_dataset(
            name="Checkerboard_Size",
            data=checkerboard_size,
            dtype="uint64",
        )

        h5_file.create_dataset(
            name="Duration_Minutes",
            data=duration_minutes,
            dtype="uint64",
        )

        h5_file.create_dataset(
            name="Shuffle",
            data=True,
            dtype="bool",
        )

    print("Finished saving stimulus.")


if __name__ == "__main__":
    generate_and_store_shuffled_l_shaped_noise(
        width_in_pixels=800,
        height_in_pixels=800,
        fps=15,
        checkerboard_size=32,
        duration_minutes=20,
        background_value=127,
        random_seed=42,
        shuffle_positions=4,
        name=Path(
            "stimuli/laura_scripts/"
            "32px_15Hz_20mins_top_left_L_shape_shuffle.h5"
        ),
    )