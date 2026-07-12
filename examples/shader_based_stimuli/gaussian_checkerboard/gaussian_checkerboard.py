import json
from pathlib import Path

import numpy as np
import moderngl

import fpspy.stim


# Get the path now, when module is loaded.
# Waiting for the stimulus program's setup() to be called could risk the current
# working directory being changed, and __file__, being relative, would point to the
# wrong place.
frag_src_path = Path(__file__).parent.resolve() / "gaussian_checkerboard.frag"


def frag_shader_src():
    with open(frag_src_path, "r") as f:
        return f.read()


class GaussianCheckerboardProgram(fpspy.stim.StimProgram):

    def __init__(
        self,
        stim_shape: tuple,
        checker_size: int,
        mean_u8: int,
        std_u8: int,
        base_seed: int,
        n_frames: int,
        fps: float,
    ) -> None:
        self.stim_shape = stim_shape
        self.checker_size = checker_size
        self.mean_u8 = mean_u8
        self.std_u8 = std_u8
        self.base_seed = base_seed
        self.n_frames = n_frames
        self.fps = fps
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
        # Each window's channels select which independent noise streams it draws.
        if channels is None:
            channels = (0, 1, 2)
        if len(channels) != 3:
            raise ValueError(f"Expected 3 channels, got {channels=}.")

        H, W = self.stim_shape

        # Compile shader program.
        vert_src = fpspy.stim.quad_vertex_shader_src()
        frag_src = frag_shader_src()
        self._program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Set static uniforms.
        self._program["resolution"].value = (W, H)
        self._program["checker_size"].value = self.checker_size
        self._program["mean_u8"].value = self.mean_u8
        self._program["std_u8"].value = self.std_u8
        self._program["base_seed"].value = self.base_seed
        self._program["channel_ids"].value = tuple(channels)

        quad = fpspy.stim.create_centered_quad(
            W,
            H,
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
        self._program["frame_num"].value = int(frame_idx)
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
        "stim_shape": (768, 768),
        "checker_size": 16,
        "mean_u8": 128,
        "std_u8": 42,
        "base_seed": 12345,
        "n_frames": 1200,
        "fps": 20.0,
    }


def default_config() -> str:
    return json.dumps(default_params(), indent=2)


def to_program(config: str) -> GaussianCheckerboardProgram:
    if config:
        params = json.loads(config)
    else:
        params = default_params()
    return GaussianCheckerboardProgram(**params)
