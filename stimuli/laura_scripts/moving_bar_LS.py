# Moving bar stimulus, but with hard step edges (not averaged/weighted). See moving_bar_area_weighted for that.
from pathlib import Path

import h5py
import hdf5plugin
import imageio.v2 as imageio
import numpy as np


# =========================================================
# USER PARAMETERS
# =========================================================

WIDTH = 800
HEIGHT = 800

FPS = 13

# Thickness of the solid black or white part
BAR_THICKNESS_PX = 20

# Distance moved on every frame
BAR_SPEED_PX_PER_FRAME = 1

# Width of the gradient on each side
RAMP_WIDTH_PX = 30

# Background luminance
BACKGROUND_VALUE = 127

# Optional grey frames between directions
INTER_DIRECTION_FRAMES = 0

OUTPUT_DIR = Path(".")

# Keep this as "Noise" if that is what your playback software expects
DATASET_NAME = "Noise"


# =========================================================
# MOTION DIRECTIONS
#
# Image coordinates:
# x increases to the right
# y increases downward
# =========================================================

DIRECTIONS = {
    "left_to_right": (1.0, 0.0),
    "right_to_left": (-1.0, 0.0),
    "bottom_to_top": (0.0, -1.0),
    "top_to_bottom": (0.0, 1.0),
    "bottom_left_to_top_right": (1.0, -1.0),
    "bottom_right_to_top_left": (-1.0, -1.0),
    "top_left_to_bottom_right": (1.0, 1.0),
    "top_right_to_bottom_left": (-1.0, 1.0),
}


# =========================================================
# COORDINATE FUNCTIONS
# =========================================================

def normalise_vector(dx, dy):
    """Convert a direction vector into a unit vector."""

    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        raise ValueError("Direction vector cannot have zero length.")

    return dx / length, dy / length


def make_coordinate_grid(width, height):
    """
    Create x and y coordinate arrays centred on the image centre.
    """

    x = (
        np.arange(width, dtype=np.float32)
        - (width - 1) / 2
    )

    y = (
        np.arange(height, dtype=np.float32)
        - (height - 1) / 2
    )

    X, Y = np.meshgrid(x, y)

    return X, Y


def projection_bounds(width, height, ux, uy):
    """
    Find the minimum and maximum coordinates of the screen along
    the chosen movement direction.
    """

    corners = np.array(
        [
            [-(width - 1) / 2, -(height - 1) / 2],
            [+(width - 1) / 2, -(height - 1) / 2],
            [-(width - 1) / 2, +(height - 1) / 2],
            [+(width - 1) / 2, +(height - 1) / 2],
        ],
        dtype=np.float32,
    )

    projections = (
        corners[:, 0] * ux
        + corners[:, 1] * uy
    )

    return projections.min(), projections.max()


def compute_front_positions(
    width,
    height,
    ux,
    uy,
    bar_thickness_px,
    ramp_width_px,
    speed_px_per_frame,
):
    """
    Calculate all bar positions needed to move the complete profile
    across the screen.

    'front' is the boundary between the solid bar and the leading ramp.

    The complete spatial profile occupies:

        front - thickness - ramp
            to
        front + ramp
    """

    if speed_px_per_frame <= 0:
        raise ValueError(
            "BAR_SPEED_PX_PER_FRAME must be greater than zero."
        )

    s_min, s_max = projection_bounds(
        width=width,
        height=height,
        ux=ux,
        uy=uy,
    )

    # At the first position, the leading edge of the complete
    # profile is just outside the screen.
    start_front = s_min - ramp_width_px

    # At the final position, the trailing edge of the complete
    # profile is just outside the opposite side.
    end_front = (
        s_max
        + bar_thickness_px
        + ramp_width_px
    )

    return np.arange(
        start_front,
        end_front + speed_px_per_frame,
        speed_px_per_frame,
        dtype=np.float32,
    )


# =========================================================
# BAR FRAME GENERATION
# =========================================================

def make_edge_bar_frame(
    projected_coordinates,
    front,
    bar_thickness_px,
    ramp_width_px,
    polarity,
    background_value=127,
):
    """
    Generate one frame of the ON-edge or OFF-edge stimulus.

    OFF profile, from trailing side to leading side:

        grey
        -> grey-to-black gradient
        -> black bar
        -> abrupt black-to-white transition
        -> white-to-grey gradient
        -> grey

    ON profile, from trailing side to leading side:

        grey
        -> grey-to-white gradient
        -> white bar
        -> abrupt white-to-black transition
        -> black-to-grey gradient
        -> grey

    Parameters
    ----------
    projected_coordinates : np.ndarray
        Coordinate of every pixel along the direction of movement.

    front : float
        Position of the boundary between the solid bar and the
        leading ramp.

    bar_thickness_px : float
        Thickness of the solid black or white bar.

    ramp_width_px : float
        Width of each gradient.

    polarity : str
        Either "ON" or "OFF".

    background_value : int
        Grey background value, normally 127.
    """

    polarity = polarity.upper()

    if polarity == "OFF":
        solid_bar_value = 0
        leading_edge_value = 255

    elif polarity == "ON":
        solid_bar_value = 255
        leading_edge_value = 0

    else:
        raise ValueError(
            "polarity must be either 'ON' or 'OFF'."
        )

    S = projected_coordinates

    frame = np.full(
        S.shape,
        fill_value=background_value,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Spatial boundaries
    # -----------------------------------------------------

    solid_end = front
    solid_start = front - bar_thickness_px

    trailing_ramp_start = solid_start - ramp_width_px
    trailing_ramp_end = solid_start

    leading_ramp_start = solid_end
    leading_ramp_end = solid_end + ramp_width_px

    # -----------------------------------------------------
    # Solid central bar
    # -----------------------------------------------------

    solid_mask = (
        (S >= solid_start)
        & (S <= solid_end)
    )

    frame[solid_mask] = solid_bar_value

    if ramp_width_px > 0:

        # =================================================
        # TRAILING RAMP
        #
        # OFF: grey -> black
        # ON:  grey -> white
        # =================================================

        trailing_mask = (
            (S >= trailing_ramp_start)
            & (S < trailing_ramp_end)
        )

        trailing_fraction = (
            S[trailing_mask] - trailing_ramp_start
        ) / ramp_width_px

        frame[trailing_mask] = (
            background_value
            + trailing_fraction
            * (solid_bar_value - background_value)
        )

        # =================================================
        # LEADING RAMP
        #
        # OFF: white -> grey
        # ON:  black -> grey
        #
        # Notice that the first point in this ramp has the
        # opposite luminance from the solid bar. This creates
        # the sharp ON or OFF edge.
        # =================================================

        leading_mask = (
            (S > leading_ramp_start)
            & (S <= leading_ramp_end)
        )

        leading_fraction = (
            S[leading_mask] - leading_ramp_start
        ) / ramp_width_px

        frame[leading_mask] = (
            leading_edge_value
            + leading_fraction
            * (background_value - leading_edge_value)
        )

    return np.clip(
        frame,
        0,
        255,
    ).astype(np.uint8)


# =========================================================
# VIDEO FUNCTION
# =========================================================

def append_video_frame(writer, frame):
    """
    Convert a greyscale frame into RGB before writing it to MP4.
    """

    rgb_frame = np.repeat(
        frame[:, :, np.newaxis],
        repeats=3,
        axis=2,
    )

    writer.append_data(rgb_frame)


# =========================================================
# STIMULUS GENERATION
# =========================================================

def generate_moving_bar_stimulus(
    polarity,
    width,
    height,
    fps,
    bar_thickness_px,
    speed_px_per_frame,
    ramp_width_px,
    background_value,
    h5_name,
    video_name,
    dataset_name="Noise",
    inter_direction_frames=0,
):
    """
    Generate all eight directions for one polarity and save them
    as both H5 and MP4 files.
    """

    X, Y = make_coordinate_grid(
        width=width,
        height=height,
    )

    direction_data = {}

    direction_names = []
    direction_start_frames = []
    direction_frame_counts = []

    total_frames = 0

    # -----------------------------------------------------
    # Prepare coordinate projections and frame counts
    # -----------------------------------------------------

    for direction_name, (dx, dy) in DIRECTIONS.items():

        ux, uy = normalise_vector(dx, dy)

        # Position of every pixel along the movement axis
        projected_coordinates = X * ux + Y * uy

        fronts = compute_front_positions(
            width=width,
            height=height,
            ux=ux,
            uy=uy,
            bar_thickness_px=bar_thickness_px,
            ramp_width_px=ramp_width_px,
            speed_px_per_frame=speed_px_per_frame,
        )

        direction_data[direction_name] = {
            "ux": ux,
            "uy": uy,
            "projected_coordinates": projected_coordinates,
            "fronts": fronts,
        }

        direction_names.append(direction_name)
        direction_start_frames.append(total_frames)
        direction_frame_counts.append(len(fronts))

        total_frames += len(fronts)

        if direction_name != list(DIRECTIONS.keys())[-1]:
            total_frames += inter_direction_frames

    # -----------------------------------------------------
    # Create H5 file
    # -----------------------------------------------------

    with h5py.File(h5_name, "w") as h5_file:

        stimulus_dataset = h5_file.create_dataset(
            dataset_name,
            shape=(total_frames, height, width),
            dtype="uint8",
            chunks=(1, height, width),
            compression=hdf5plugin.Blosc(
                cname="blosclz",
                clevel=9,
                shuffle=hdf5plugin.Blosc.NOSHUFFLE,
            ),
        )

        # -------------------------------------------------
        # Open video writer
        # -------------------------------------------------

        with imageio.get_writer(
            video_name,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        ) as video_writer:

            frame_index = 0

            for direction_number, direction_name in enumerate(
                DIRECTIONS
            ):

                projected_coordinates = (
                    direction_data[direction_name][
                        "projected_coordinates"
                    ]
                )

                fronts = direction_data[direction_name]["fronts"]

                print(
                    f"Generating {polarity} stimulus: "
                    f"{direction_name} "
                    f"({len(fronts)} frames)"
                )

                for front in fronts:

                    frame = make_edge_bar_frame(
                        projected_coordinates=projected_coordinates,
                        front=front,
                        bar_thickness_px=bar_thickness_px,
                        ramp_width_px=ramp_width_px,
                        polarity=polarity,
                        background_value=background_value,
                    )

                    stimulus_dataset[frame_index] = frame

                    append_video_frame(
                        writer=video_writer,
                        frame=frame,
                    )

                    frame_index += 1

                # Grey pause between directions
                is_last_direction = (
                    direction_number
                    == len(DIRECTIONS) - 1
                )

                if (
                    inter_direction_frames > 0
                    and not is_last_direction
                ):

                    blank_frame = np.full(
                        (height, width),
                        fill_value=background_value,
                        dtype=np.uint8,
                    )

                    for _ in range(inter_direction_frames):

                        stimulus_dataset[frame_index] = blank_frame

                        append_video_frame(
                            writer=video_writer,
                            frame=blank_frame,
                        )

                        frame_index += 1

        # -------------------------------------------------
        # H5 metadata
        # -------------------------------------------------

        h5_file.create_dataset(
            name="Frame_Rate",
            data=fps,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Stimulus_Width",
            data=width,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Stimulus_Height",
            data=height,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Bar_Thickness_Px",
            data=bar_thickness_px,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Bar_Speed_Px_Per_Frame",
            data=speed_px_per_frame,
            dtype="float32",
        )

        h5_file.create_dataset(
            name="Ramp_Width_Px",
            data=ramp_width_px,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Background_Value",
            data=background_value,
            dtype="uint8",
        )

        h5_file.create_dataset(
            name="Inter_Direction_Frames",
            data=inter_direction_frames,
            dtype="uint16",
        )

        h5_file.create_dataset(
            name="Polarity",
            data=np.bytes_(polarity),
        )

        h5_file.create_dataset(
            name="Directions",
            data=np.asarray(
                direction_names,
                dtype="S",
            ),
        )

        h5_file.create_dataset(
            name="Direction_Start_Frame",
            data=np.asarray(
                direction_start_frames,
                dtype=np.int32,
            ),
        )

        h5_file.create_dataset(
            name="Direction_Frame_Count",
            data=np.asarray(
                direction_frame_counts,
                dtype=np.int32,
            ),
        )

    print()
    print(f"Saved H5: {h5_name}")
    print(f"Saved video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"Total stimulus time (mins): {(total_frames/FPS)/60}")
    print()


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_moving_bar_stimulus(
        polarity="ON",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        bar_thickness_px=BAR_THICKNESS_PX,
        speed_px_per_frame=BAR_SPEED_PX_PER_FRAME,
        ramp_width_px=RAMP_WIDTH_PX,
        background_value=BACKGROUND_VALUE,
        h5_name=OUTPUT_DIR / "moving_bar_ON.h5",
        video_name=OUTPUT_DIR / "moving_bar_ON.mp4",
        dataset_name=DATASET_NAME,
        inter_direction_frames=INTER_DIRECTION_FRAMES,
    )

    generate_moving_bar_stimulus(
        polarity="OFF",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        bar_thickness_px=BAR_THICKNESS_PX,
        speed_px_per_frame=BAR_SPEED_PX_PER_FRAME,
        ramp_width_px=RAMP_WIDTH_PX,
        background_value=BACKGROUND_VALUE,
        h5_name=OUTPUT_DIR / "moving_bar_OFF.h5",
        video_name=OUTPUT_DIR / "moving_bar_OFF.mp4",
        dataset_name=DATASET_NAME,
        inter_direction_frames=INTER_DIRECTION_FRAMES,
    )


if __name__ == "__main__":
    main()