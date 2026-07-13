#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    // Map from clip space [-1,1] to texture coords [0,1]
    uv = in_pos * 0.5 + 0.5;
}
