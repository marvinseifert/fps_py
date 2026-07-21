import moderngl
from typing import Optional
import numpy as np
import logging
import fpspy.color
import importlib
import json

_logger = logging.getLogger(__name__)


COLOR_ENCODINGS = {"identity", "srgb", "ti-video-enhanced"}

def make_encode_lut(encoding: str, n: int = 1024) -> np.ndarray:
    """Build a lookup table mapping linear [0,1] to display input values [0,1].

    Parameters
    ----------
    encoding : str
        - "identity": no mapping to be applied.
        - "srgb"
        - "ti-video-enhanced": inverse of the DLP4500's "TI Video (Enhanced)"
          curve, so that displayed intensity ends up linear in our values.
    n : int
        Number of table entries.

    Returns
    -------
    np.ndarray
        (n,) float32, lut[i] = display input for linear value i/(n-1).
    """
    if encoding not in COLOR_ENCODINGS:
        raise ValueError(
            f"Unsupported encoding '{encoding}'. Supported: {COLOR_ENCODINGS}"
        )
    ramp = np.linspace(0.0, 1.0, n)
    if encoding == "identity":
        lut = ramp
    elif encoding == "srgb":
        lut = fpspy.color.to_srgb(ramp)
    elif encoding == "ti-video-enhanced":
        path = (
            importlib.resources.files("fpspy.resources")
            / "ti_video_enhanced_2-5-0-0.json"
        )
        with (path).open("r") as f:
            table = json.load(f)
        xs = np.asarray(table["x"])  # display input
        ys = np.asarray(table["y"])  # displayed intensity (linear)
        if np.any(np.diff(xs) < 0) or np.any(np.diff(ys) < 0):
            raise ValueError(f"Gamma table {path} not monotonic.")
        # Invert the forward table: for each linear target, the input producing it.
        lut = np.interp(ramp, ys, xs)
    else:
        assert False, f"Unsupported encoding {encoding}"
    return lut.astype(np.float32)



class DisplayAdapter:
    """GPU post-processing pass adapting stimulus output for a display.

    Runs decode -> intensity correction -> clip -> encode (see module docstring).
    Intensity correction is skipped by passing intensity_map=None; the other
    stages are configured by `is_stim_linear`, `clip_mode` and `encoding`.

    Usage:
        adapter = DisplayAdapter(ctx, width, height, intensity_map, encoding)
        # In render loop:
        adapter.fbo.use()            # render stimulus into the FBO
        ... render stimulus ...
        adapter.render(ctx)           # draw adapted result to current framebuffer
        # Cleanup:
        adapter.release()
    """

    # Post-processing shaders for the display-adaptation pass.
    VERTEX_SHADER = "display_adapter_vertex_shader.glsl"
    FRAG_SHADER = "display_adapter_fragment_shader.glsl"

    STIM_TEX_UNIT = 0
    IMAP_TEX_UNIT = 1
    LUT_TEX_UNIT = 2

    CLIP_MODES = {"pixel_rescale": 0, "clip": 1, "none": 2}

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        encoding: str,
        intensity_map: Optional[np.ndarray] = None,
        intensity_prescale: float = 1.0,
        clip_mode: str = "pixel_rescale",
        mirror: bool = False,
        rotation: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Parameters
        ----------
        ctx : moderngl.Context
        width, height : int
            Framebuffer dimensions (should match window size).
        intensity_map : np.ndarray or None
            (H, W, 3) float32 array in [0, 1], where 1.0 = full intensity.
            H, W should match the display dimensions. None disables intensity
            correction (a 1x1 map of ones is used, making the division a no-op).
        encoding : str
            The display's transfer curve. See make_encode_lut.
        intensity_prescale : float
            Multiplier applied before the intensity division. Values < 1.0
            leave room for the correction to boost dim regions without
            clipping past 1.0.
        clip_mode : str
            How to handle corrected values > 1.0:
            - "pixel_rescale": scale all channels by max channel (preserves hue)
            - "clip": hard clamp to [0, 1]
            - "none": pass through unclamped (encode still clamps to [0, 1])
        logger:
            If you want the DisplayAdapter to use the presenter's logger, pass it
            in. Otherwise, it will use the module's logger.
        """
        if clip_mode not in self.CLIP_MODES:
            raise ValueError(f"clip_mode must be one of {list(self.CLIP_MODES)}")
        self.encoding = encoding
        self.intensity_prescale = intensity_prescale
        self.clip_mode = clip_mode
        self.logger = logger or _logger

        # FBO for rendering the stimulus into.
        self.stim_texture = ctx.texture((width, height), 4, dtype="f2")
        self.stim_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = ctx.framebuffer(color_attachments=[self.stim_texture])

        # Upload intensity map as a texture.
        self.imap_texture = self._make_intensity_map_texture(
            ctx, width, height, intensity_map, mirror, rotation
        )
        # Upload the encoding curve as an (N, 1) lookup-table texture.
        lut = make_encode_lut(encoding)
        self.lut_texture = ctx.texture((len(lut), 1), 1, data=lut.tobytes(), dtype="f4")
        self.lut_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lut_texture.repeat_x = False
        self.lut_texture.repeat_y = False

        # Compile the post-processing shader.
        vertex_shader_path = (
            importlib.resources.files("fpspy.resources") / self.VERTEX_SHADER
        )
        frag_shader_path = (
            importlib.resources.files("fpspy.resources") / self.FRAG_SHADER
        )
        self.program = ctx.program(
            vertex_shader=vertex_shader_path.read_text(),
            fragment_shader=frag_shader_path.read_text(),
        )
        self.logger.info(
            f"DisplayAdapter shader compiled.\n"
            f"\tvertex shader: {vertex_shader_path}\n"
            f"\tfragment shader: {frag_shader_path}"
        )
        self.program["stimulus_tex"].value = self.STIM_TEX_UNIT
        self.program["intensity_map_tex"].value = self.IMAP_TEX_UNIT
        self.program["encode_lut_tex"].value = self.LUT_TEX_UNIT
        self.program["encode_lut_n"].value = len(lut)
        self.program["intensity_prescale"].value = self.intensity_prescale
        self.program["clip_mode"].value = self.CLIP_MODES[clip_mode]

        # Fullscreen quad (two triangles covering clip space).
        # fmt: off
        quad_verts = np.array(
            [#    x   y
                [-1, -1],  # bottom left
                [-1,  1],  # top left
                [ 1, -1],  # bottom right
                [ 1, -1],  # bottom right
                [-1,  1],  # top left
                [ 1,  1],  # top right
            ],
            dtype=np.float32,
        )
        # fmt: on
        self.vbo = ctx.buffer(quad_verts.tobytes())
        self.vao = ctx.simple_vertex_array(self.program, self.vbo, "in_pos")

    @staticmethod
    def _make_intensity_map_texture(
        ctx, width, height, intensity_map, mirror, rotation
    ):
        """Apply the window's mirror/rotation to the intensity map and upload it.

        The transforms match the visual effect of the stimulus quad transform
        (create_centered_quad): mirror is a horizontal flip, and a positive
        rotation is counter-clockwise on screen; mirror is applied first.
        Pixel-exact numpy flips, so only right-angle rotations are supported.
        """
        if intensity_map is None:
            # No correction: a single white texel divides as a no-op.
            # As color encoding only adapters are possible, we should still make an
            # intensity map (identity), so that we don't need a switch in the shader.
            imap = np.ones((1, 1, 3), dtype=np.float32)
        else:
            imap = intensity_map
            if mirror:
                imap = np.fliplr(imap)
            if rotation % 90 != 0:
                raise ValueError(
                    "Only right-angle rotations are supported for the "
                    f"intensity map. Got: {rotation}"
                )
            imap = np.rot90(imap, int(rotation // 90) % 4)
            if imap.shape[:2] != (height, width):
                raise ValueError(
                    "Intensity map must match the window size after "
                    f"mirror/rotation. Map (H, W): {imap.shape[:2]}, "
                    f"window (H, W): {(height, width)}."
                )
        h, w, c = imap.shape
        if c != 3:
            raise ValueError(
                f"Intensity map must have 3 channels (RGB). Got {c} channels."
            )
        # OpenGL expects bottom-to-top row order.
        imap = np.ascontiguousarray(imap[::-1])
        imap_texture = ctx.texture((w, h), c, data=imap.tobytes(), dtype="f4")
        imap_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        imap_texture.repeat_x = False
        imap_texture.repeat_y = False
        return imap_texture

    def render(self, ctx: moderngl.Context):
        """Draw the adapted stimulus to the currently bound framebuffer.

        Call this after rendering the stimulus into self.fbo.
        """
        self.stim_texture.use(location=self.STIM_TEX_UNIT)
        self.imap_texture.use(location=self.IMAP_TEX_UNIT)
        self.lut_texture.use(location=self.LUT_TEX_UNIT)
        self.vao.render(moderngl.TRIANGLES)

    def release(self):
        """Release all GPU resources."""
        for resource in [
            self.vao,
            self.vbo,
            self.stim_texture,
            self.imap_texture,
            self.lut_texture,
            self.fbo,
        ]:
            if resource is not None:
                resource.release()