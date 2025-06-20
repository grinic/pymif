import dask.array as da
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
    data_levels: da.Array,
    metadata: Dict[str, Any],
    num_levels: int = 3,
    downscale_factor: int = 2,
    start_level: int = 0,
) -> Tuple[List[da.Array], Dict[str, Any]]:
    """
    Generate a multiscale pyramid and updated metadata for NGFF-compatible OME-Zarr writing.

    Parameters
    ----------
        base_level : da.Array
            Dask array of the highest resolution (shape: TCZYX).
        metadata : Dict[str, Any]
            Metadata dict with at least 'scales' and 'axes'.
        num_levels : int
            Total number of pyramid levels to generate.
        downscale_factor : int
            Factor by which to reduce spatial dims at each level.
        start_level : int
            Use this level from existing pyramid (if any) as base level.

    Returns
    -------
        Tuple[List[da.Array], Dict[str, Any]]
            tuple containing the pyramidal dataset and the updated metadata
    """
    if data_levels[0].ndim != 5:
        raise ValueError("Expected base_level to have shape (T, C, Z, Y, X)")

    print(f"Requested start level {start_level}")
    if start_level < len(data_levels):
        # Enough preexisting levels — just slice from start_level
        print("Resolution layer already available")
        pyramid = [data_levels[start_level]]
        new_scales = [metadata["scales"][start_level]]

    else:
        # Need to compute base_level from highest available level
        print("Resoluytion layer not available: will compute")
        base_downscale = downscale_factor ** start_level
        current = pad_to_divisible(data_levels[0], [base_downscale] * 3)
        down = downsample_nn(current, [base_downscale] * 3)
        pyramid = [down]
        new_scales = [(i*base_downscale for i in metadata["scales"][0])]
            
    for _ in range(1, num_levels):
        current = pad_to_divisible(pyramid[-1], [downscale_factor] * 3)
        down = downsample_nn(current, [downscale_factor] * 3)
        pyramid.append(down)

    # Update metadata scales:
    for level in range(1,num_levels):
        scale_factor = downscale_factor ** level
        new_scales.append(tuple(s * scale_factor for s in new_scales[0]))

    metadata["scales"] = new_scales

    # Update metadata size:
    new_size = []
    for level in range(num_levels):
        new_size.append(pyramid[level].shape)

    metadata["size"] = new_size

    return pyramid, metadata

