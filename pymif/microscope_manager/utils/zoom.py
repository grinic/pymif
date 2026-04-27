import dask.array as da
import numpy as np
import scipy

from .downsampling import SpatialFactor, normalize_spatial_factor


def _zoom_numpy(arr: np.ndarray, scale: SpatialFactor) -> np.ndarray:
    """Nearest-neighbor zoom/shrink on spatial ZYX axes."""
    from scipy.ndimage import zoom

    spatial_scale = normalize_spatial_factor(
        scale,
        name="scale",
        allow_float=True,
    )

    factors = [1.0] * (arr.ndim - 3) + list(spatial_scale)

    for axis, factor in enumerate(factors):
        if arr.shape[axis] * factor < 1:
            factors[axis] = 1.0

    return zoom(arr, zoom=factors, order=0)


def _zoom_dask(arr: da.Array, scale: SpatialFactor) -> da.Array:
    """Nearest-neighbor zoom/shrink on spatial ZYX axes for dask arrays."""
    if not isinstance(arr, da.Array):
        arr = da.from_array(arr, chunks="auto")

    if arr.ndim < 3:
        raise ValueError("Array must have at least 3 dimensions: Z, Y, X.")

    spatial_scale = normalize_spatial_factor(
        scale,
        name="scale",
        allow_float=True,
    )

    factors = [1.0] * (arr.ndim - 3) + list(spatial_scale)

    for axis, factor in enumerate(factors):
        if arr.shape[axis] * factor < 1:
            factors[axis] = 1.0

    def _zoom_block(block, zoom):
        return scipy.ndimage.zoom(block, zoom=zoom, order=0)

    out_chunks = tuple(
        tuple(max(1, int(round(chunk * factor))) for chunk in axis_chunks)
        for axis_chunks, factor in zip(arr.chunks, factors)
    )

    return arr.map_blocks(
        _zoom_block,
        zoom=factors,
        dtype=arr.dtype,
        chunks=out_chunks,
    )