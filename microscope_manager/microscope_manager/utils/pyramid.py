import dask.array as da
import numpy as np
from typing import List, Tuple, Dict, Any

def pad_to_divisible(array: da.Array, factors: List[int]) -> da.Array:
    """Pad spatial axes (Z, Y, X) so they are divisible by the corresponding downscale factors."""
    pad_width = [(0, 0)] * array.ndim
    for i, f in zip([-3, -2, -1], factors):  # assuming ZYX are last 3 dims
        size = array.shape[i]
        remainder = size % f
        if remainder != 0:
            pad_width[i] = (0, f - remainder)
    return da.pad(array, pad_width, mode="edge")

def downsample_nn(array: da.Array, factors: List[int]) -> da.Array:
    """Nearest-neighbor downsampling via striding in ZYX."""
    slicing = [slice(None)] * array.ndim
    for i, f in zip([-3, -2, -1], factors):
        slicing[i] = slice(0, None, f)
    return array[tuple(slicing)]

def build_pyramid(
    base_level: da.Array,
    metadata: Dict[str, Any],
    num_levels: int = 3,
    downscale_factor: int = 2,
) -> Tuple[List[da.Array], Dict[str, Any]]:
    """
    Generate a multiscale pyramid and updated metadata for NGFF-compatible OME-Zarr writing.

    Parameters:
    - base_level: Dask array of the highest resolution (shape: TCZYX).
    - metadata: Metadata dict with at least 'scales' and 'axes'.
    - num_levels: Total number of pyramid levels to generate.
    - downscale_factor: Factor by which to reduce spatial dims at each level.

    Returns:
    - (pyramid_levels, updated_metadata)
    """
    if base_level.ndim != 5:
        raise ValueError("Expected base_level to have shape (T, C, Z, Y, X)")

    pyramid = [base_level]
    for _ in range(1, num_levels):
        current = pad_to_divisible(pyramid[-1], [downscale_factor] * 3)
        down = downsample_nn(current, [downscale_factor] * 3)
        pyramid.append(down)

    # Update metadata scales:
    scales = metadata["scales"]
    new_scales = []
    for level in range(num_levels):
        scale_factor = downscale_factor ** level
        new_scales.append(tuple(s * scale_factor for s in scales[0]))

    metadata["scales"] = new_scales

    return pyramid, metadata

