import moderngl
import moderngl_window
from moderngl_window.conf import settings
import numpy as np
import time
import pyglet

# Get the current display
display = pyglet.canvas.get_display()

# Get the list of screens
screens = display.get_screens()

checkerboard_size = 100
freq = 30
frames = 5*60*freq
# Configure the window settings
settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'
settings.WINDOW['gl_version'] = (4, 1)
settings.WINDOW['size'] = (1000, 1000)  # Adjust this to your desired window size
settings.WINDOW['aspect_ratio'] = 1000/ 1000
settings.WINDOW["fullscreen"] = False


# Check if there are at least two screens (primary and secondary)



# Create the window using the configured settings
window = moderngl_window.create_window_from_settings()
window.position = (2560, 0)
width_in_pixels = window.width
height_in_pixels = window.height
aspect_adjustment = width_in_pixels / height_in_pixels

# Create a ModernGL context
ctx = moderngl.create_context()

# Vertex Shader
vertex_shader = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_pos * 0.5 + 0.5;  // Convert position range [-1, 1] to uv range [0, 1]
}
"""

# Fragment Shader
fragment_shader = """
#version 330

uniform sampler2D pattern;
out vec4 out_color;
in vec2 uv;
uniform float aspect_adjustment;

void main() {
    // Adjust the UV coordinates
    vec2 adjusted_uv = vec2(uv.x * aspect_adjustment, uv.y);
    float value = texture(pattern, adjusted_uv).r;
    out_color = vec4(value, value, value, 1.0);
}
"""

# Create program
program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

# Screen quad vertices
# Full screen quad vertices
quad = np.array([
    -1.0,  1.0,  # top left
    -1.0, -1.0,  # bottom left
     1.0,  1.0,  # top right
     1.0,  1.0,  # top right
    -1.0, -1.0,  # bottom left
     1.0, -1.0   # bottom right
], dtype=np.float32)

vbo = ctx.buffer(quad.tobytes())
vao = ctx.simple_vertex_array(program, vbo, 'in_pos')


def generate_checkerboard_pattern(checker_size, width_in_pixels, height_in_pixels):
    # Calculate the number of squares based on pixel resolution
    pattern_width = width_in_pixels // checker_size
    pattern_height = height_in_pixels // checker_size

    pattern_shape = (pattern_height, pattern_width)
    pattern = np.random.randint(0, 2, pattern_shape, dtype=np.uint8) * 255
    pattern_texture = np.repeat(np.repeat(pattern, checker_size, axis=0), checker_size, axis=1)

    return pattern_texture

# Create textures for patterns
patterns = [ctx.texture((width_in_pixels, height_in_pixels), 1, pattern.tobytes()) for pattern in
            [generate_checkerboard_pattern(checkerboard_size, width_in_pixels, height_in_pixels) for _ in range(frames)]]




current_pattern_index = 0

desired_fps = freq  # Set frame rate to 20 Hz
time_per_frame = 1.0 / desired_fps
last_update = time.time()
current_pattern_index = 0

while not window.is_closing:


    window.use()  # Ensure the correct context is being used
    start_time = time.time()  # Start time for this frame
    elapsed_time = start_time - last_update

    if elapsed_time >= time_per_frame:
        # Clear the window
        ctx.clear(1.0, 1.0, 1.0)

        # Bind the texture to texture unit 0
        patterns[current_pattern_index].use(location=0)

        # Set the shader uniform to the index of the texture unit
        program['aspect_adjustment'].value = aspect_adjustment

        program['pattern'].value = 0

        # Render the full screen quad
        vao.render(moderngl.TRIANGLES)

        # Swap buffers after rendering
        window.swap_buffers()

        # Measure frame duration
        frame_duration = time.time() - start_time
        if frame_duration > time_per_frame:
            print(
                f"WARNING: Frame duration of {frame_duration * 1000:.2f} ms exceeds the desired {time_per_frame * 1000:.2f} ms!")

        # Move to the next pattern
        current_pattern_index = (current_pattern_index + 1) % len(patterns)

        # Update the last update time
        last_update = start_time
    else:
        # Sleep for a short duration to avoid busy waiting
        time.sleep(0.001)  # Sleep for 1 millisecond

