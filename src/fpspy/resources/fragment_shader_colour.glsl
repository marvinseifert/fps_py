#version 330

uniform sampler2D tex;
out vec4 out_color;
in vec2 uv;
//uniform float aspect_adjustment;
uniform vec2 scale;
void main() {

    float red = texture(tex, uv).r;
    float green = texture(tex, uv).g;
    float blue = texture(tex, uv).b;
    float alpha = texture(tex, uv).a;

    out_color = vec4(red, green, blue, alpha);
}

