#version 330

uniform sampler2D pattern;
out vec4 out_color;
in vec2 uv;
//uniform float aspect_adjustment;
uniform vec2 scale;
void main() {

    float red = texture(pattern, uv).r;
    float green = texture(pattern, uv).g;
    float blue = texture(pattern, uv).b;
    float alpha = texture(pattern, uv).a;

    out_color = vec4(red, green, blue, alpha);
}

