import OpenGL.GL as gl


def probe_default_fbo_srgb() -> tuple[bool, bool]:
    """Check if the default framebuffer has sRGB enabled.

    ModernGL is opaque about whether the default framebuffer is sRGB-capable, so we
    check manually.
    """
    # Are we converting linear->sRGB on framebuffer writes?
    srgb_enabled = bool(gl.glIsEnabled(gl.GL_FRAMEBUFFER_SRGB))

    # Is the default framebuffer attachment sRGB-encoded or linear?
    enc = gl.glGetFramebufferAttachmentParameteriv(
        gl.GL_FRAMEBUFFER,
        gl.GL_BACK_LEFT,  # default framebuffer color buffer
        gl.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
    )
    # enc is an int enum: GL_SRGB or GL_LINEAR
    srgb_capable = int(enc) == int(gl.GL_SRGB)
    return srgb_capable, srgb_enabled