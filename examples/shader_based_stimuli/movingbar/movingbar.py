import json
import math
import numpy as np
import moderngl
from pathlib import Path
import fpspy.stim


# Get the path now, when module is loaded.
# Waitin for the stimulus program's setup() to be called could risk the current working
# directory being changed, and __file__, being relative, would point to the wrong place.

frag_src_path = Path(__file__).parent.resolve() / "movingbar.frag"

def frag_shader_src():
    with open(frag_src_path, "r") as f:
        return f.read()



class MovingBarProgram(fpspy.stim.StimProgram):

    def __init__(
        self,
        stim_shape: tuple,
        n_channels: int,
        thickness: float,
        n_directions: int,
        speed: float,
        start_buffer_px: float,
        fps: float,
    ) -> None:
        self.stim_shape = stim_shape
        self.n_channels = n_channels
        self.thickness = thickness
        self.n_directions = n_directions
        self.speed = speed
        self.start_buffer_px = start_buffer_px
        self.fps = fps
        # Compute radius and frame counts from stim_shape, not window size.
        H, W = self.stim_shape
        tk_half = self.thickness / 2
        self.radius = math.sqrt(H**2 + W**2) / 2 + self.start_buffer_px + tk_half
        self.frames_per_direction = math.ceil((2 * self.radius) / self.speed)
        self.n_frames = self.frames_per_direction * self.n_directions
        self._setup_done = False

    def setup(
        self,
        ctx,
        win_width,
        win_height,
        channels=None,
        mirror=None,
        rotation=None,
        win_id=None,
    ):
        if self._setup_done:
            raise RuntimeError("setup() has already been called.")
        if channels is None:
            raise ValueError("MovingBarProgram requires channels to be specified.")

        self.W = win_width
        self.H = win_height

        # Compile shader program.
        vert_src = fpspy.stim.quad_vertex_shader_src()
        frag_src = frag_shader_src()
        self._program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Set static uniforms.
        self._program["resolution"].value = (self.W, self.H)
        self._program["thickness"].value = self.thickness

        # Fullscreen quad (1x1 stim broadcasts to full window).
        quad = fpspy.stim.create_centered_quad(
            1,
            1,
            win_width,
            win_height,
            mirror=mirror or False,
            rotation=rotation or 0.0,
        )
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.simple_vertex_array(self._program, self._vbo, "in_pos")

        self._setup_done = True

        frame_times = np.arange(self.n_frames + 1) / self.fps
        triggers = np.arange(self.n_frames, dtype=np.int64)
        return frame_times, triggers

    def render(self, ctx, frame_idx, global_frame_num):
        # Which direction and how far along.
        direction_idx = frame_idx // self.frames_per_direction
        frame_within = frame_idx % self.frames_per_direction

        theta = direction_idx * 2 * math.pi / self.n_directions
        bar_position = frame_within * self.speed - self.radius

        self._program["theta"].value = theta
        self._program["bar_position"].value = bar_position
        self._vao.render(moderngl.TRIANGLES)

    def cleanup(self) -> None:
        if hasattr(self, "_vao") and self._vao is not None:
            self._vao.release()
        if hasattr(self, "_vbo") and self._vbo is not None:
            self._vbo.release()
        self._vao = None
        self._vbo = None


def default_params() -> dict:
    return {
        "stim_shape": (500, 500),
        "n_channels": 6,
        "thickness": 20.0,
        "n_directions": 8,
        "speed": 2.0,
        "start_buffer_px": 5.0,
        "fps": 50.0,
    }


def default_config() -> str:
    return json.dumps(default_params())


def to_program(config: str) -> MovingBarProgram:
    if config:
        params = json.loads(config)
    else:
        params = default_params()
    return MovingBarProgram(**params)
