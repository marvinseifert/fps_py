#version 330

uniform sampler2D tex;
out vec4 out_color;
in vec2 uv;
//uniform float aspect_adjustment;
uniform vec2 scale;
void main() {

    float value = texture(tex, uv).r;

    out_color = vec4(value, value, value, 1.0);
}

