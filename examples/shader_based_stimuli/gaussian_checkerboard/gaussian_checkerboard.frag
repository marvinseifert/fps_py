#version 330

// Deterministic Gaussian-ish checkerboard noise (CLT approx) for GLSL 330.
// - No log/sqrt/sin/cos.
// - Integer-only RNG based on MurmurHash3 fmix32 (standard finalizer).
// - CLT: sum of 12 uniforms -> approx N(0,1) after normalization.
//
// Outputs RGBA in [0,1] but computed from uint8 values for stability.

out vec4 out_color;
in vec2 uv;

uniform ivec2 resolution;     // (width, height)
uniform int checker_size;      // pixels per checker (must be > 0)
uniform uint base_seed;        // global seed
uniform uint frame_num;        // updated each frame

// Mean/std expressed in 8-bit units for determinism and safe integer scaling.
// mean_u8 in [0..255], std_u8 in [0..255]. Typical: mean_u8=128.
// std_u8 controls contrast. Keep modest to avoid clipping.
uniform int mean_u8;
uniform int std_u8;

// Which noise streams this window's R, G, B outputs draw from. Windows showing
// different channel sets (e.g. two DLPs covering channels 0-2 and 3-5) get
// independent noise.
uniform ivec3 channel_ids;

// --- MurmurHash3 fmix32 (public-domain style finalizer) ---
uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Build a 32-bit "counter" from structured keys.
// You can change constants, but keep it linear + fmix32.
// This is not ad-hoc mixing; fmix32 is doing the heavy lifting.
uint make_counter(uint frame, uint cx, uint cy, uint channel, uint sample_idx, uint seed) {
    uint x = seed;
    x ^= frame * 0x9e3779b9u;
    x ^= cx    * 0x85ebca6bu;
    x ^= cy    * 0xc2b2ae35u;
    x ^= channel * 0x27d4eb2du;
    x ^= sample_idx * 0x165667b1u;
    return x;
}

// Uniform uint32 -> uniform-ish uint16 (top 16 bits after fmix).
uint u16_from_u32(uint x) {
    return (fmix32(x) >> 16) & 0xffffu;
}

// Approx Gaussian: sum of 12 U(0,1) - 6 has mean 0, var 1.
// We do it in uint16, then scale by 65535 to avoid float dependency.
// Let u_i ~ Uniform{0..65535}. Then E[u]=32767.5.
// s = sum(u_i) - 12*32768 is approximately centered and has std ~ 65535.
int approx_gauss_q16(uint frame, uint cx, uint cy, uint channel, uint seed) {
    // 12 samples
    uint sum = 0u;
    for (uint i = 0u; i < 12u; i++) {
        uint ctr = make_counter(frame, cx, cy, channel, i, seed);
        sum += u16_from_u32(ctr);
    }

    // Center around 0; using 32768 keeps it integer and symmetric.
    // Range roughly [-393216, +393216].
    int centered = int(sum) - (12 * 32768);
    return centered;
}

// Convert centered ~N(0, 65535^2) into an 8-bit intensity with mean/std control.
// intensity = mean_u8 + std_u8 * centered / 65535
// All integer math; deterministic.
int intensity_u8(uint frame, uint cx, uint cy, uint channel, uint seed) {
    int g = approx_gauss_q16(frame, cx, cy, channel, seed);

    // Scale: std_u8 * g / 65535 fits in int32 since std_u8 <= 255 and |g| <= ~4e5
    // product <= ~1e8.
    int scaled = (std_u8 * g) / 65535;

    int v = mean_u8 + scaled;

    // Hard clip (still deterministic). Choose mean/std to make clipping rare.
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return v;
}

void main() {
    // Pixel coords
    vec2 pix = uv * vec2(resolution);

    // Checker coords
    uint cx = uint(int(pix.x) / checker_size);
    uint cy = uint(int(pix.y) / checker_size);

    // Independent RGB streams.
    int r8 = intensity_u8(frame_num, cx, cy, uint(channel_ids.x), base_seed);
    int g8 = intensity_u8(frame_num, cx, cy, uint(channel_ids.y), base_seed);
    int b8 = intensity_u8(frame_num, cx, cy, uint(channel_ids.z), base_seed);

    vec3 rgb = vec3(r8, g8, b8) * (1.0 / 255.0);
    out_color = vec4(rgb, 1.0);
}
