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