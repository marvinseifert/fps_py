#version 330

in vec2 in_pos;
uniform vec2 scale;
out vec2 uv;

void main() {
    gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);

    // Define UV coordinates based on vertex index
    switch (gl_VertexID % 6) {
        case 0: uv = vec2(0.0, 1.0); break; // top-left
        case 1: uv = vec2(0.0, 0.0); break; // bottom-left
        case 2: uv = vec2(1.0, 1.0); break; // top-right
        case 3: uv = vec2(1.0, 1.0); break; // top-right (duplicate for second triangle)
        case 4: uv = vec2(0.0, 0.0); break; // bottom-left (duplicate for second triangle)
        case 5: uv = vec2(1.0, 0.0); break; // bottom-right
    }
     //= uv.x*scale.x;
    //uv.y = uv.y/scale.y;
    //uv.x = uv.x/scale.x;
}


//void main() {
//    gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);
//
//    //gl_Position = vec4(in_pos*scale, 0.0, 1.0);
//    uv = in_pos * 0.5 + 0.5;  // Convert position range [-1, 1] to uv range [0, 1]
//    //uv.y /= scale.y;
//}
