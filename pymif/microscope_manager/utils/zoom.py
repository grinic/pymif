import numpy as np
import dask.array as da
import scipy

def _zoom_numpy(arr: np.ndarray, scale: float) -> np.ndarray:
    """Simple 2x zoom or shrink using nearest-neighbor."""
    from scipy.ndimage import zoom
    factors = [1] * (arr.ndim - 3) + [scale, scale, scale]
    for i, f in enumerate(factors):
        if (arr.shape[i]*f) < 1:
            factors[i] = 1
    return zoom(arr, zoom=factors, order=0)


def _zoom_dask(arr: da.Array, scale: float) -> da.Array:
    """Zoom for dask arrays."""
    if not isinstance(arr, da.Array):
        arr = da.from_array(arr, chunks="auto")

    ndim = arr.ndim
    if ndim < 3:
        raise ValueError("Array must have at least 3 dimensions (Z, Y, X)")

    # Build zoom factors: 1 for non-spatial axes, scale for last 3 axes
    factors = [1] * (ndim - 3) + [scale, scale, scale]

    # Make sure zoom doesn't reduce any dimension below 1
    for i, f in enumerate(factors):
        if arr.shape[i] * f < 1:
            factors[i] = 1

    # Apply zoom blockwise
    def _zoom_block(block, zoom):
        return scipy.ndimage.zoom(block, zoom=zoom, order=0)

    result = arr.map_blocks(
        _zoom_block,
        zoom=factors,
        dtype=arr.dtype
    )

    return result
