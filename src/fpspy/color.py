import numpy as np


def to_srgb(arr):
    """Scale linear RGB values to sRGB.

    sRGB conversion doesn't specify a datatype, scaling or quantization,
    so use to_srgb8 if you need to get 8-bit values in [0, 255].

    Args:
        arr: Input array with linear RGB values in [0, 1].
    Returns:
        Array with sRGB values in [0, 1] as float32.
    """
    # shape = arr.shape
    # flat_arr = arr.reshape(-1, 3)
    # srgb_flat = colour.cctf_encoding(flat_arr, function='sRGB')
    # srgb = srgb_flat.reshape(shape)
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Input array values must be in the range [0, 1].")
    arr = np.where(
        arr <= 0.0031308, arr * 12.92, 1.055 * (arr**(1 / 2.4)) - 0.055
    )
    return arr


def to_srgb8(arr):
    """Scale linear RGB values [0, 1] to sRGB 8-bit [0, 255].

    Args:
        arr: Input array with linear RGB values in [0, 1].
    Returns:
        Array with sRGB values in [0, 255] as uint8.
    """
    srgb = to_srgb(arr)
    srgb8 = np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
    return srgb8


