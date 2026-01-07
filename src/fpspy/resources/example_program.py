import fpspy.stim
import moderngl
import importlib.resources
import numpy as np


class MyProgram(fpspy.stim.StimProgram):

    # We always bind to texture unit 0.
    TEXTURE_UNIT = 0

    def __init__(self) -> None:
        self.textures = None
        self.single_tex = None
        self.win_id = None
        self._setup_done = False
        self.W = 500
        self.H = 500
        self.F = 1000
        self.fps = 60

    def setup(self, ctx, win_width, win_height, channels=None, win_id=None):
        if self._setup_done:
            raise RuntimeError("TextureSequence.setup() has already been called.")
        self.win_id = win_id

        # Compile program and load vertices.
        self._program = self._compile_program(ctx)
        quad = fpspy.stim.create_centered_quad(self.W, self.H, win_width, win_height)
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.simple_vertex_array(self._program, self._vbo, "in_pos")
        self._setup_done = True
        frame_times = 1.0 / self.fps * np.arange(self.F)
        triggers = np.arange(self.F, dtype=np.int64)
        return frame_times, triggers

    def _compile_program(self, ctx):
        """Read shaders and compile the program."""
        resource_dir = importlib.resources.files("fpspy.resources")
        with (resource_dir / fpspy.stim.QUAD_VERTEX_SHADER).open("r") as vertex_file:
            vert_src = vertex_file.read()
        with (resource_dir / fpspy.stim.TEXTURE_FRAG_SHADER).open("r") as fragment_file:
            frag_src = fragment_file.read()
        program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        program["tex"].value = self.TEXTURE_UNIT
        return program

    def render(self, ctx, frame_idx, global_frame_num):
        """
        Design note: we could pass global_frame_num only, and have this class decide
        what to do. That would make it have to know about looping. Or, we could only
        pass frame_idx, assumed to be looped, and don't pass global_frame_num at all.
        """
        # TODO: zoom!
        # Create texture on demand.
        if self.single_tex is not None:
            self.single_tex.release()
        self.single_tex = ctx.texture(
            (self.W, self.H),
            3,
            np.random.default_rng()
            .integers(0, 256, (self.H, self.W, 3), dtype=np.uint8)
            .tobytes(),
            samples=0,
            alignment=1,
        )
        # Use nearest-neighbor filtering for crisp pixels when zoomed.
        self.single_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.single_tex.use(location=self.TEXTURE_UNIT)

        self._vao.render(moderngl.TRIANGLES)

    def cleanup(self) -> None:
        """Release OpenGL resources."""
        if self.single_tex is not None:
            self.single_tex.release()
        self.single_tex = None
        if self._vao is not None:
            self._vao.release()
        if self._vbo is not None:
            self._vbo.release()
        self._vao = None
        self._vbo = None


def to_program(config: str):
    return MyProgram()
