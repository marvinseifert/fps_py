#version 330

in vec2 in_pos;
uniform vec2 scale;
out vec2 uv;

void main() {
    vec2 pos = in_pos;

    gl_Position = vec4(pos, 0.0, 1.0);

    // UV coordinates - flipped V to match StimArray's (0,0)=top-left convention
    // without needing a CPU-side row-reverse before each texture upload.
    switch (gl_VertexID % 6) {
        case 0: uv = vec2(0.0, 0.0); break; // top-left
        case 1: uv = vec2(0.0, 1.0); break; // bottom-left
        case 2: uv = vec2(1.0, 0.0); break; // top-right
        case 3: uv = vec2(1.0, 0.0); break; // top-right
        case 4: uv = vec2(0.0, 1.0); break; // bottom-left
        case 5: uv = vec2(1.0, 1.0); break; // bottom-right
    }
}


