#version 330
in vec2 in_pos;
out vec2 uv;
uniform vec2 scale;
uniform float aspect_adjustment;
void main() {
    gl_Position = vec4(in_pos.x * scale.x, in_pos.y * scale.y, 0.0, 1.0);

    //gl_Position = vec4(in_pos*scale, 0.0, 1.0);
    uv = in_pos * 0.5 + 0.5;  // Convert position range [-1, 1] to uv range [0, 1]
}
