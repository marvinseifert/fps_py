## Moving edge stimulus (ON and OFF separately) where edge of bar pixels are weighted by their bar coverage to prevent hard steps along edge of bar

from pathlib import Path

import h5py
import imageio.v2 as imageio

try:
    import hdf5plugin
except ModuleNotFoundError:
    hdf5plugin = None
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# =========================================================
# USER PARAMETERS
# =========================================================

WIDTH = 800
HEIGHT = 800

FPS = 13        ## this should cover ~5 cone rows per second, which should be sufficient to sample cone rows individually

# Thickness of the solid black or white part.
# Fractional pixel values are allowed.
BAR_THICKNESS_PX = 20

# Distance moved on every frame.
# Fractional pixel values are allowed.
BAR_SPEED_PX_PER_FRAME = 1.0

# Width of the gradient on each side.
# Fractional pixel values are allowed.
RAMP_WIDTH_PX = 30.0

# Background luminance.
BACKGROUND_VALUE = 127

# Optional grey frames between directions.
INTER_DIRECTION_FRAMES = 0

OUTPUT_DIR = Path(".")

# Keep this as "Noise" if that is what your playback software expects.
DATASET_NAME = "Noise"

# Plot and save explanatory anti-aliasing schematics.
MAKE_SCHEMATICS = True
SCHEMATIC_NAME = "moving_bar_area_weighting_schematic.png"

# Generate the full H5 and MP4 stimuli.
GENERATE_STIMULI = True




def h5_compression_kwargs():
    """Return Blosc compression settings, or a portable gzip fallback."""

    if hdf5plugin is not None:
        return {
            "compression": hdf5plugin.Blosc(
                cname="blosclz",
                clevel=9,
                shuffle=hdf5plugin.Blosc.NOSHUFFLE,
            )
        }

    print(
        "Warning: hdf5plugin is not installed; using gzip compression. "
        "Install hdf5plugin to reproduce the original Blosc format."
    )
    return {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }


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
    Create x and y arrays containing the centres of the projector pixels.

    For an even-sized 800 x 800 image, the x and y pixel centres run from
    -399.5 to +399.5. Each projector pixel is treated as a 1 x 1 square
    extending 0.5 pixel either side of its centre.
    """

    x = np.arange(width, dtype=np.float64) - (width - 1) / 2
    y = np.arange(height, dtype=np.float64) - (height - 1) / 2

    return np.meshgrid(x, y)



def projection_bounds(width, height, ux, uy):
    """
    Find the minimum and maximum pixel-centre coordinates along the
    chosen movement direction.
    """

    corners = np.array(
        [
            [-(width - 1) / 2, -(height - 1) / 2],
            [+(width - 1) / 2, -(height - 1) / 2],
            [-(width - 1) / 2, +(height - 1) / 2],
            [+(width - 1) / 2, +(height - 1) / 2],
        ],
        dtype=np.float64,
    )

    projections = corners[:, 0] * ux + corners[:, 1] * uy

    return projections.min(), projections.max()



def pixel_projection_half_width(ux, uy):
    """
    Maximum distance, along the movement axis, from the centre of a
    1 x 1 projector pixel to any point inside that pixel.

    This is 0.5 px for a horizontal/vertical edge and sqrt(2)/2 px for
    a 45-degree edge.
    """

    return 0.5 * (abs(ux) + abs(uy))



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
    Calculate all bar positions required to move the complete continuous
    profile across the screen.

    ``front`` is the sharp boundary between the solid bar and the leading
    ramp. The complete profile occupies:

        front - bar_thickness - ramp_width
            to
        front + ramp_width

    Whole motion steps are added before and after the positions used in the
    original script. This preserves the original subpixel phase while making
    the first and last frames completely background, including every part of
    edge pixels rather than only their centre points.
    """

    if speed_px_per_frame <= 0:
        raise ValueError("BAR_SPEED_PX_PER_FRAME must be greater than zero.")

    if bar_thickness_px <= 0:
        raise ValueError("BAR_THICKNESS_PX must be greater than zero.")

    if ramp_width_px < 0:
        raise ValueError("RAMP_WIDTH_PX cannot be negative.")

    s_min, s_max = projection_bounds(
        width=width,
        height=height,
        ux=ux,
        uy=uy,
    )

    half_pixel_projection = pixel_projection_half_width(ux, uy)

    # Preserve the subpixel phase used by the original script, while adding
    # a whole number of motion steps before and after it. This means the
    # anti-aliased edge appears at the same positions as before, but the
    # first and last frames contain background only.
    original_start_front = s_min - ramp_width_px
    original_end_front = (
        s_max + bar_thickness_px + ramp_width_px
    )

    extra_steps = int(
        np.ceil(half_pixel_projection / speed_px_per_frame)
    )

    start_front = (
        original_start_front
        - extra_steps * speed_px_per_frame
    )
    end_front = (
        original_end_front
        + extra_steps * speed_px_per_frame
    )

    return np.arange(
        start_front,
        end_front + speed_px_per_frame,
        speed_px_per_frame,
        dtype=np.float64,
    )


# =========================================================
# IDEAL CONTINUOUS BAR PROFILE
# =========================================================


def polarity_values(polarity):
    """Return the solid-bar and leading-edge luminances."""

    polarity = polarity.upper()

    if polarity == "OFF":
        return 0.0, 255.0

    if polarity == "ON":
        return 255.0, 0.0

    raise ValueError("polarity must be either 'ON' or 'OFF'.")



def evaluate_ideal_profile(
    relative_coordinate,
    bar_thickness_px,
    ramp_width_px,
    polarity,
    background_value,
):
    """
    Evaluate the ideal continuous luminance profile at one or more points.

    ``relative_coordinate`` is the projected position relative to ``front``:

        q = projected_coordinate - front

    This function samples at mathematical points. It does not yet account
    for the finite area of a projector pixel.
    """

    q = np.asarray(relative_coordinate, dtype=np.float64)
    solid_value, leading_value = polarity_values(polarity)

    values = np.full(q.shape, float(background_value), dtype=np.float64)

    if ramp_width_px > 0:
        trailing_start = -(bar_thickness_px + ramp_width_px)
        trailing_end = -bar_thickness_px

        trailing_mask = (q >= trailing_start) & (q < trailing_end)
        trailing_fraction = (
            q[trailing_mask] - trailing_start
        ) / ramp_width_px

        values[trailing_mask] = (
            background_value
            + trailing_fraction * (solid_value - background_value)
        )

    solid_mask = (q >= -bar_thickness_px) & (q < 0.0)
    values[solid_mask] = solid_value

    if ramp_width_px > 0:
        leading_mask = (q >= 0.0) & (q <= ramp_width_px)
        leading_fraction = q[leading_mask] / ramp_width_px

        values[leading_mask] = (
            leading_value
            + leading_fraction * (background_value - leading_value)
        )

    return values


# =========================================================
# EXACT PIXEL-AREA AVERAGING
# =========================================================


def projection_cdf_and_first_moment(values, ux, uy):
    """
    Calculate the distribution of positions within one square pixel after
    projection onto the movement axis.

    Let a subpixel point have offsets dx and dy from the pixel centre, with
    both uniformly distributed from -0.5 to +0.5. Its displacement along
    the movement axis is:

        U = ux * dx + uy * dy

    This function returns:

        F(t) = P(U <= t)
        M(t) = E[U * I(U <= t)]

    The formula is exact. For horizontal/vertical motion, U is uniformly
    distributed. For diagonal motion it is triangular; for an arbitrary
    angle it is trapezoidal.
    """

    t = np.asarray(values, dtype=np.float64)

    a = 0.5 * abs(float(ux))
    b = 0.5 * abs(float(uy))

    large_half_width = max(a, b)
    small_half_width = min(a, b)

    if large_half_width == 0:
        raise ValueError("The movement direction cannot have zero length.")

    cdf = np.zeros_like(t)
    first_moment = np.zeros_like(t)

    # Horizontal or vertical motion: uniform distribution.
    if small_half_width < 1e-15:
        lower = -large_half_width
        upper = +large_half_width

        middle = (t > lower) & (t < upper)
        above = t >= upper

        cdf[middle] = (
            t[middle] + large_half_width
        ) / (2.0 * large_half_width)
        cdf[above] = 1.0

        first_moment[middle] = (
            t[middle] ** 2 - large_half_width**2
        ) / (4.0 * large_half_width)

        return cdf, first_moment

    # General angle: convolution of two uniform distributions.
    A = large_half_width
    B = small_half_width

    support = A + B
    plateau_half_width = A - B

    rising = (t > -support) & (t < -plateau_half_width)
    plateau = (
        (t >= -plateau_half_width)
        & (t <= plateau_half_width)
    )
    falling = (t > plateau_half_width) & (t < support)
    above = t >= support

    v = t[rising] + support

    cdf[rising] = v**2 / (8.0 * A * B)
    first_moment[rising] = (
        v**3 / 3.0 - support * v**2 / 2.0
    ) / (4.0 * A * B)

    v_at_plateau_start = 2.0 * B
    moment_at_plateau_start = (
        v_at_plateau_start**3 / 3.0
        - support * v_at_plateau_start**2 / 2.0
    ) / (4.0 * A * B)

    cdf[plateau] = (t[plateau] + A) / (2.0 * A)
    first_moment[plateau] = (
        moment_at_plateau_start
        + (
            t[plateau] ** 2
            - plateau_half_width**2
        ) / (4.0 * A)
    )

    w = support - t[falling]

    cdf[falling] = 1.0 - w**2 / (8.0 * A * B)
    first_moment[falling] = -(
        support * w**2 / 2.0 - w**3 / 3.0
    ) / (4.0 * A * B)

    cdf[above] = 1.0

    return cdf, first_moment



def interval_probability_and_moment(
    pixel_centre_coordinate,
    lower_edge,
    upper_edge,
    ux,
    uy,
):
    """
    For each pixel centre q, calculate:

        P(lower_edge <= q + U < upper_edge)

    and the first moment of U over that interval.
    """

    q = np.asarray(pixel_centre_coordinate, dtype=np.float64)

    upper_cdf, upper_moment = projection_cdf_and_first_moment(
        upper_edge - q,
        ux,
        uy,
    )
    lower_cdf, lower_moment = projection_cdf_and_first_moment(
        lower_edge - q,
        ux,
        uy,
    )

    probability = upper_cdf - lower_cdf
    moment = upper_moment - lower_moment

    return probability, moment



def exact_pixel_average_profile(
    relative_pixel_centres,
    ux,
    uy,
    bar_thickness_px,
    ramp_width_px,
    polarity,
    background_value,
):
    """
    Calculate the exact area-averaged luminance of each projector pixel.

    The ideal bar profile is piecewise linear. The average of a linear
    function over the part of a square pixel lying inside each profile
    section can therefore be calculated exactly from the projected area
    distribution and its first moment.

    Consequently, a pixel cut 20% by a white region and 80% by a black
    region is assigned approximately:

        0.20 * 255 + 0.80 * 0 = 51

    rather than being forced to either 0 or 255 according to its centre.
    """

    q = np.asarray(relative_pixel_centres, dtype=np.float64)
    background = float(background_value)
    solid_value, leading_value = polarity_values(polarity)

    result = np.full(q.shape, background, dtype=np.float64)

    # -----------------------------------------------------
    # Trailing ramp: background -> solid bar
    # -----------------------------------------------------

    if ramp_width_px > 0:
        lower = -(bar_thickness_px + ramp_width_px)
        upper = -bar_thickness_px

        slope = (solid_value - background) / ramp_width_px
        intercept = background - slope * lower

        probability, moment = interval_probability_and_moment(
            pixel_centre_coordinate=q,
            lower_edge=lower,
            upper_edge=upper,
            ux=ux,
            uy=uy,
        )

        result += (
            (intercept + slope * q - background) * probability
            + slope * moment
        )

    # -----------------------------------------------------
    # Solid central bar
    # -----------------------------------------------------

    probability, _ = interval_probability_and_moment(
        pixel_centre_coordinate=q,
        lower_edge=-bar_thickness_px,
        upper_edge=0.0,
        ux=ux,
        uy=uy,
    )

    result += (solid_value - background) * probability

    # -----------------------------------------------------
    # Leading ramp: opposite edge value -> background
    # -----------------------------------------------------

    if ramp_width_px > 0:
        lower = 0.0
        upper = ramp_width_px

        slope = (background - leading_value) / ramp_width_px
        intercept = leading_value

        probability, moment = interval_probability_and_moment(
            pixel_centre_coordinate=q,
            lower_edge=lower,
            upper_edge=upper,
            ux=ux,
            uy=uy,
        )

        result += (
            (intercept + slope * q - background) * probability
            + slope * moment
        )

    return np.clip(result, 0.0, 255.0)



def profile_breakpoints(bar_thickness_px, ramp_width_px):
    """Positions where the ideal profile changes value or slope."""

    if ramp_width_px > 0:
        return np.array(
            [
                -(bar_thickness_px + ramp_width_px),
                -bar_thickness_px,
                0.0,
                ramp_width_px,
            ],
            dtype=np.float64,
        )

    return np.array(
        [-bar_thickness_px, 0.0],
        dtype=np.float64,
    )


# =========================================================
# BAR FRAME GENERATION
# =========================================================


def make_edge_bar_frame_centre_sampled(
    projected_coordinates,
    front,
    bar_thickness_px,
    ramp_width_px,
    polarity,
    background_value=127,
):
    """
    Old centre-sampled behaviour, retained only for schematic comparison.
    """

    relative_coordinate = projected_coordinates - front

    frame = evaluate_ideal_profile(
        relative_coordinate=relative_coordinate,
        bar_thickness_px=bar_thickness_px,
        ramp_width_px=ramp_width_px,
        polarity=polarity,
        background_value=background_value,
    )

    return np.rint(np.clip(frame, 0, 255)).astype(np.uint8)



def make_edge_bar_frame(
    projected_coordinates,
    front,
    ux,
    uy,
    bar_thickness_px,
    ramp_width_px,
    polarity,
    background_value=127,
):
    """
    Generate one anti-aliased ON-edge or OFF-edge frame.

    Unlike the previous implementation, each projector pixel is treated as
    a square area rather than a single point at its centre. Pixels wholly
    inside one linear section use their centre value, which is already the
    exact area average. Only pixels crossed by a profile boundary need the
    more detailed area calculation.

    OFF profile, trailing side to leading side:

        grey -> grey-to-black -> black
             -> sharp black-to-white edge
             -> white-to-grey -> grey

    ON profile, trailing side to leading side:

        grey -> grey-to-white -> white
             -> sharp white-to-black edge
             -> black-to-grey -> grey
    """

    relative_coordinate = projected_coordinates - front

    # Centre sampling is exact wherever the whole pixel lies inside a
    # constant or linear section of the profile.
    frame = evaluate_ideal_profile(
        relative_coordinate=relative_coordinate,
        bar_thickness_px=bar_thickness_px,
        ramp_width_px=ramp_width_px,
        polarity=polarity,
        background_value=background_value,
    )

    # Only pixels whose square area crosses a breakpoint need correction.
    half_width = pixel_projection_half_width(ux, uy)
    breakpoints = profile_breakpoints(
        bar_thickness_px=bar_thickness_px,
        ramp_width_px=ramp_width_px,
    )

    crossed_boundary = np.zeros(frame.shape, dtype=bool)

    for breakpoint in breakpoints:
        crossed_boundary |= (
            np.abs(relative_coordinate - breakpoint)
            < half_width + 1e-12
        )

    if np.any(crossed_boundary):
        frame[crossed_boundary] = exact_pixel_average_profile(
            relative_pixel_centres=relative_coordinate[crossed_boundary],
            ux=ux,
            uy=uy,
            bar_thickness_px=bar_thickness_px,
            ramp_width_px=ramp_width_px,
            polarity=polarity,
            background_value=background_value,
        )

    # Round to the nearest projector input level rather than truncating.
    return np.rint(np.clip(frame, 0, 255)).astype(np.uint8)


# =========================================================
# SCHEMATIC PLOTS
# =========================================================


def binary_edge_white_fraction(pixel_centres, edge_position, ux, uy):
    """
    Exact fraction of each pixel lying on the white side of the ideal edge:

        ux*x + uy*y <= edge_position
    """

    cdf, _ = projection_cdf_and_first_moment(
        edge_position - pixel_centres,
        ux,
        uy,
    )

    return cdf



def plot_antialiasing_schematics(output_path):
    """
    Save a four-panel schematic explaining the area-weighting method.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # -----------------------------------------------------
    # A. Diagonal ideal edge and percentage coverage
    # -----------------------------------------------------

    ax = axes[0, 0]

    size = 7
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2
    X, Y = np.meshgrid(coords, coords)

    ux, uy = normalise_vector(1.0, 1.0)
    projected = X * ux + Y * uy
    edge_position = 0.18

    white_fraction = binary_edge_white_fraction(
        pixel_centres=projected,
        edge_position=edge_position,
        ux=ux,
        uy=uy,
    )

    ax.imshow(
        white_fraction,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[-3.5, 3.5, -3.5, 3.5],
        interpolation="nearest",
    )

    for y in coords:
        for x in coords:
            row = int(y + (size - 1) / 2)
            col = int(x + (size - 1) / 2)
            value = white_fraction[row, col]

            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="tab:red",
                    linewidth=0.8,
                )
            )

            text_colour = "white" if value < 0.45 else "black"
            ax.text(
                x,
                y,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=text_colour,
            )

    x_line = np.linspace(-4.0, 4.0, 200)
    y_line = (edge_position - ux * x_line) / uy
    ax.plot(x_line, y_line, color="tab:blue", linewidth=2.2)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title(
        "A. Ideal diagonal edge crossing square projector pixels\n"
        "Labels show the exact white-area fraction"
    )
    ax.set_xlabel("x position (pixels)")
    ax.set_ylabel("y position (pixels)")

    # -----------------------------------------------------
    # B. One-dimensional hard edge
    # -----------------------------------------------------

    ax = axes[0, 1]

    centres = np.arange(-5, 6, dtype=np.float64)
    edge_position = 0.25

    centre_sampled = (centres <= edge_position).astype(float)
    area_weighted = binary_edge_white_fraction(
        pixel_centres=centres,
        edge_position=edge_position,
        ux=1.0,
        uy=0.0,
    )

    bar_width = 0.36
    ax.bar(
        centres - bar_width / 2,
        centre_sampled,
        width=bar_width,
        label="Old: value at pixel centre",
        alpha=0.75,
    )
    ax.bar(
        centres + bar_width / 2,
        area_weighted,
        width=bar_width,
        label="New: mean over pixel area",
        alpha=0.75,
    )

    ax.axvline(
        edge_position,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ideal edge",
    )

    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(centres)
    ax.set_xlabel("Pixel-centre position")
    ax.set_ylabel("White fraction / normalised luminance")
    ax.set_title(
        "B. A crossed pixel receives an intermediate luminance\n"
        "rather than being classified only by its centre"
    )
    ax.legend(loc="lower left")

    # -----------------------------------------------------
    # C and D. Actual ON profile before and after correction
    # -----------------------------------------------------

    demo_size = 70
    X_demo, Y_demo = make_coordinate_grid(demo_size, demo_size)
    ux_demo, uy_demo = normalise_vector(1.0, 1.0)
    projected_demo = X_demo * ux_demo + Y_demo * uy_demo

    front = 4.17
    thickness = 10.0
    ramp = 13.0

    old_frame = make_edge_bar_frame_centre_sampled(
        projected_coordinates=projected_demo,
        front=front,
        bar_thickness_px=thickness,
        ramp_width_px=ramp,
        polarity="ON",
        background_value=BACKGROUND_VALUE,
    )

    new_frame = make_edge_bar_frame(
        projected_coordinates=projected_demo,
        front=front,
        ux=ux_demo,
        uy=uy_demo,
        bar_thickness_px=thickness,
        ramp_width_px=ramp,
        polarity="ON",
        background_value=BACKGROUND_VALUE,
    )

    zoom = np.s_[15:55, 15:55]

    ax = axes[1, 0]
    ax.imshow(
        old_frame[zoom],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax.set_title(
        "C. Previous centre-sampled frame\n"
        "Diagonal boundaries form hard stair steps"
    )
    ax.set_xlabel("Projector pixel")
    ax.set_ylabel("Projector pixel")

    ax = axes[1, 1]
    image = ax.imshow(
        new_frame[zoom],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax.set_title(
        "D. Area-weighted frame\n"
        "Boundary pixels encode the fraction of each ideal region"
    )
    ax.set_xlabel("Projector pixel")
    ax.set_ylabel("Projector pixel")

    colourbar_axis = fig.add_axes([0.27, 0.035, 0.46, 0.018])
    colourbar = fig.colorbar(
        image,
        cax=colourbar_axis,
        orientation="horizontal",
    )
    colourbar.set_label("8-bit luminance")

    fig.suptitle(
        "Pixel-area anti-aliasing of the ideal moving-bar stimulus",
        fontsize=17,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.11,
        top=0.93,
        wspace=0.24,
        hspace=0.28,
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved schematic: {output_path}")


# =========================================================
# VIDEO FUNCTION
# =========================================================


def append_video_frame(writer, frame):
    """Convert a greyscale frame into RGB before writing it to MP4."""

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
    Generate all eight directions for one polarity and save them as H5
    and MP4 files.
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
    last_direction_name = list(DIRECTIONS)[-1]

    # -----------------------------------------------------
    # Prepare coordinate projections and frame counts
    # -----------------------------------------------------

    for direction_name, (dx, dy) in DIRECTIONS.items():
        ux, uy = normalise_vector(dx, dy)

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

        if direction_name != last_direction_name:
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
            **h5_compression_kwargs(),
        )

        with imageio.get_writer(
            video_name,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        ) as video_writer:
            frame_index = 0

            for direction_number, direction_name in enumerate(DIRECTIONS):
                data = direction_data[direction_name]
                projected_coordinates = data["projected_coordinates"]
                fronts = data["fronts"]
                ux = data["ux"]
                uy = data["uy"]

                print(
                    f"Generating {polarity} stimulus: "
                    f"{direction_name} "
                    f"({len(fronts)} frames)"
                )

                for front in fronts:
                    frame = make_edge_bar_frame(
                        projected_coordinates=projected_coordinates,
                        front=front,
                        ux=ux,
                        uy=uy,
                        bar_thickness_px=bar_thickness_px,
                        ramp_width_px=ramp_width_px,
                        polarity=polarity,
                        background_value=background_value,
                    )

                    stimulus_dataset[frame_index] = frame
                    append_video_frame(video_writer, frame)
                    frame_index += 1

                is_last_direction = (
                    direction_number == len(DIRECTIONS) - 1
                )

                if inter_direction_frames > 0 and not is_last_direction:
                    blank_frame = np.full(
                        (height, width),
                        fill_value=background_value,
                        dtype=np.uint8,
                    )

                    for _ in range(inter_direction_frames):
                        stimulus_dataset[frame_index] = blank_frame
                        append_video_frame(video_writer, blank_frame)
                        frame_index += 1

        # -------------------------------------------------
        # H5 metadata
        # -------------------------------------------------

        h5_file.create_dataset("Frame_Rate", data=fps, dtype="uint8")
        h5_file.create_dataset("Stimulus_Width", data=width, dtype="uint16")
        h5_file.create_dataset("Stimulus_Height", data=height, dtype="uint16")
        h5_file.create_dataset(name="Checkerboard_Size", data=1, dtype="uint64")
        h5_file.create_dataset(name="Shuffle", data=False, dtype="bool")
        h5_file.create_dataset(
            "Bar_Thickness_Px",
            data=bar_thickness_px,
            dtype="float32",
        )
        h5_file.create_dataset(
            "Bar_Speed_Px_Per_Frame",
            data=speed_px_per_frame,
            dtype="float32",
        )
        h5_file.create_dataset(
            "Ramp_Width_Px",
            data=ramp_width_px,
            dtype="float32",
        )
        h5_file.create_dataset(
            "Background_Value",
            data=background_value,
            dtype="uint8",
        )
        h5_file.create_dataset(
            "Inter_Direction_Frames",
            data=inter_direction_frames,
            dtype="uint16",
        )
        h5_file.create_dataset("Polarity", data=np.bytes_(polarity))
        h5_file.create_dataset(
            "Directions",
            data=np.asarray(direction_names, dtype="S"),
        )
        h5_file.create_dataset(
            "Direction_Start_Frame",
            data=np.asarray(direction_start_frames, dtype=np.int32),
        )
        h5_file.create_dataset(
            "Direction_Frame_Count",
            data=np.asarray(direction_frame_counts, dtype=np.int32),
        )
        h5_file.create_dataset(
            "Pixel_Area_Antialiasing",
            data=True,
            dtype="bool",
        )
        h5_file.create_dataset(
            "Antialiasing_Method",
            data=np.bytes_(
                "Exact mean of the continuous piecewise-linear profile "
                "over each 1x1 square projector pixel"
            ),
        )

    print()
    print(f"Saved H5: {h5_name}")
    print(f"Saved video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"Total stimulus time (mins): {(total_frames / fps) / 60}")
    print()


# =========================================================
# MAIN
# =========================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MAKE_SCHEMATICS:
        plot_antialiasing_schematics(
            OUTPUT_DIR / SCHEMATIC_NAME
        )

    if not GENERATE_STIMULI:
        return

    generate_moving_bar_stimulus(
        polarity="ON",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        bar_thickness_px=BAR_THICKNESS_PX,
        speed_px_per_frame=BAR_SPEED_PX_PER_FRAME,
        ramp_width_px=RAMP_WIDTH_PX,
        background_value=BACKGROUND_VALUE,
        h5_name=OUTPUT_DIR / f"moving_bar_weighted_{FPS}fps_ON_{WIDTH}px_wide_{HEIGHT}px_high_{BAR_THICKNESS_PX}px_thick_{RAMP_WIDTH_PX}px_ramp_{BAR_SPEED_PX_PER_FRAME}px_per_frame.h5",
        video_name=OUTPUT_DIR / "moving_bar_ON_area_weighted.mp4",
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
        h5_name=OUTPUT_DIR / f"moving_edge_weighted_{FPS}fps_OFF_{WIDTH}px_wide_{HEIGHT}px_high__{BAR_THICKNESS_PX}px_thick_{RAMP_WIDTH_PX}px_ramp_{BAR_SPEED_PX_PER_FRAME}px_per_frame.h5",
        video_name=OUTPUT_DIR / "moving_edge_OFF_area_weighted.mp4",
        dataset_name=DATASET_NAME,
        inter_direction_frames=INTER_DIRECTION_FRAMES,
    )


if __name__ == "__main__":
    main()

