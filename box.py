import moderngl
import moderngl_window
from moderngl_window.conf import settings
import struct

settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'
settings.WINDOW['gl_version'] = (4, 1)
settings.WINDOW['size'] = (1000, 1000)  # Adjust this to your desired window size
settings.WINDOW['aspect_ratio'] = 1000/ 1000
settings.WINDOW["fullscreen"] = False

# Create a window
window = moderngl_window.create_window_from_settings()

# Create a moderngl context
ctx = moderngl.create_context()
ctx.disable(moderngl.DEPTH_TEST)
ctx.viewport = (0, 0, window.width, window.height)
# Compile the shaders
prog = ctx.program(vertex_shader=open('box_vertex.glsl').read(), fragment_shader=open('box_fragment.glsl').read())

# Define the box vertices
box = ctx.buffer(b"""
    -1.0  1.0
     1.0  1.0
     1.0 -1.0
    -1.0 -1.0
""")

# Define the box indices
ibo = ctx.buffer(b"\0\1\2\2\3\0")

triangle = ctx.buffer(b"""
    -0.5  -0.5
     0.5  -0.5
     0.0   0.5
""")

# Create a Vertex Array Object for the triangle
vao = ctx.simple_vertex_array(prog, triangle, 'in_vert')

# Create a Vertex Array Object for the box
#vao = ctx.simple_vertex_array(prog, box, 'in_vert', index_buffer=ibo)

# Model matrix (scale to 200x200)
scale_factor = 200.0 / window.width  # Assuming the window's coordinate system is -1 to 1
model = [
    [scale_factor, 0.0, 0.0, 0.0],
    [0.0, scale_factor, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]
flattened_model = [item for sublist in model for item in sublist]
model_bytes = struct.pack('16f', *flattened_model)
prog['model'].write(model_bytes)
if not prog:
    print("Error in shader compilation:", prog.error)
    exit()
# Projection matrix (identity for this example)
projection = [
    [2.0/window.width, 0.0, 0.0, -1.0],
    [0.0, 2.0/window.height, 0.0, -1.0],
    [0.0, 0.0, -2.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]
flattened_projection = [item for sublist in projection for item in sublist]
projection_bytes = struct.pack('16f', *flattened_projection)
prog['projection'].write(projection_bytes)

while not window.is_closing:
    ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear the background to black
    vao.render(moderngl.TRIANGLES, vertices=3)  # Render the box
    window.swap_buffers()
window.close()

