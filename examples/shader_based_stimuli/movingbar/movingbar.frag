#version 330

// Moving bar stimulus - procedural fragment shader.
// Renders a single smooth-edged bar moving in a given direction.

out vec4 out_color;
in vec2 uv;

uniform ivec2 resolution;      // (width, height) in pixels
uniform float thickness;        // bar thickness in pixels
uniform float theta;            // current direction angle in radians
uniform float bar_position;     // bar center position along direction axis (pixels from center)

void main() {
    // Pixel coordinates, centered on the screen.
    vec2 pix = uv * vec2(resolution) - vec2(resolution) / 2.0;

    vec2 dir = vec2(cos(theta), sin(theta));
    float projected = dot(pix, dir);
    float dist = abs(projected - bar_position);

    // Smoothstep anti-aliased edges: 1 pixel transition zone.
    float half_tk = thickness / 2.0;
    float value = smoothstep(half_tk + 1.0, half_tk - 1.0, dist);

    out_color = vec4(value, value, value, 1.0);
}
