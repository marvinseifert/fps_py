#version 330

in vec2 in_pos;
uniform vec2 scale;
uniform bool u_mirror;      // Mirror horizontally (applied first)
uniform float u_rotation;   // Rotation in degrees (0, 90, 180, 270)
out vec2 uv;

void main() {
    vec2 pos = in_pos;

    // Apply mirror to vertex position
    if (u_mirror) {
        pos.x = -pos.x;
    }

    // Apply rotation to vertex position (around origin)
    if (u_rotation != 0.0) {
        float angle = radians(u_rotation);
        float c = cos(angle);
        float s = sin(angle);
        pos = vec2(c * pos.x - s * pos.y,
                   s * pos.x + c * pos.y);
    }

    gl_Position = vec4(pos, 0.0, 1.0);

    // UV coordinates - simple corner-to-corner mapping
    switch (gl_VertexID % 6) {
        case 0: uv = vec2(0.0, 1.0); break; // top-left
        case 1: uv = vec2(0.0, 0.0); break; // bottom-left
        case 2: uv = vec2(1.0, 1.0); break; // top-right
        case 3: uv = vec2(1.0, 1.0); break; // top-right
        case 4: uv = vec2(0.0, 0.0); break; // bottom-left
        case 5: uv = vec2(1.0, 0.0); break; // bottom-right
    }
}


