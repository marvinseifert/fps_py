#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D stimulus_tex;
uniform sampler2D intensity_map_tex;
// (N, 1) table mapping linear [0,1] to display input values [0,1]. The display's
// transfer curve (sRGB, DLP4500 "enhanced", identity, ...) is baked into this table
// on the CPU; the shader has a single encode path.
uniform sampler2D encode_lut_tex;
uniform int encode_lut_n;
uniform float intensity_prescale;
// 0 = pixel_rescale, 1 = clip, 2 = none
uniform int clip_mode;

// sRGB -> linear conversion (single channel).
float srgb_to_linear(float c) {
    return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}
vec3 srgb_to_linear_v(vec3 c) {
    return vec3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
}

float encode(float v) {
    // Sample at texel centers: v=0 hits the center of texel 0 and v=1 the center of
    // texel N-1, so the table endpoints are honored exactly. Linear filtering between
    // texel centers interpolates the table, like np.interp.
    float u = (clamp(v, 0.0, 1.0) * float(encode_lut_n - 1) + 0.5)
              / float(encode_lut_n);
    return texture(encode_lut_tex, vec2(u, 0.5)).r;
}
vec3 encode_v(vec3 c) {
    return vec3(encode(c.r), encode(c.g), encode(c.b));
}

void main() {
    vec3 stim = texture(stimulus_tex, uv).rgb;
    vec3 intensity = texture(intensity_map_tex, uv).rgb;

	// Not used. Assumed to be linear.
    //// Convert to linear if the stimulus outputs sRGB.
    // vec3 stim = is_stim_linear ? stim : srgb_to_linear_v(stim);

    // Divide by intensity map (linear) to correct non-uniform illumination.
    intensity = max(intensity, vec3(0.01));
    vec3 corrected = stim * intensity_prescale / intensity;

    // Handle values > 1.0 according to clip_mode.
    if (clip_mode == 0) {
        // pixel_rescale: preserve hue by scaling all channels equally.
        float max_ch = max(corrected.r, max(corrected.g, corrected.b));
        if (max_ch > 1.0) {
            corrected /= max_ch;
        }
    } else if (clip_mode == 1) {
        // clip: hard clamp to [0, 1].
        corrected = clamp(corrected, 0.0, 1.0);
    }
    // clip_mode == 2: none. Values outside [0,1] are still clamped by encode(),
    // as a lookup table has nothing to say about them.

    frag_color = vec4(encode_v(corrected), 1.0);
}
