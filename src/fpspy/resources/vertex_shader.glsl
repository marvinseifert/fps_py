#version 330

in vec2 in_pos;
uniform vec2 scale;
uniform bool u_mirror;      // Mirror horizontally (applied first)
uniform float u_rotation;   // Rotation in degrees (0, 90, 180, 270)
out vec2 uv;

void main() {
    gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);

    // Define UV coordinates based on vertex index
    vec2 base_uv;
    switch (gl_VertexID % 6) {
        case 0: base_uv = vec2(0.0, 1.0); break; // top-left
        case 1: base_uv = vec2(0.0, 0.0); break; // bottom-left
        case 2: base_uv = vec2(1.0, 1.0); break; // top-right
        case 3: base_uv = vec2(1.0, 1.0); break; // top-right (duplicate for second triangle)
        case 4: base_uv = vec2(0.0, 0.0); break; // bottom-left (duplicate for second triangle)
        case 5: base_uv = vec2(1.0, 0.0); break; // bottom-right
    }

    // Apply mirror horizontally first
    if (u_mirror) {
        base_uv.x = 1.0 - base_uv.x;
    }

    // Apply rotation around center (0.5, 0.5)
    if (u_rotation != 0.0) {
        float angle = radians(u_rotation);
        float c = cos(angle);
        float s = sin(angle);
        base_uv -= 0.5;
        base_uv = vec2(c * base_uv.x - s * base_uv.y,
                       s * base_uv.x + c * base_uv.y);
        base_uv += 0.5;
    }

    uv = base_uv;
}


