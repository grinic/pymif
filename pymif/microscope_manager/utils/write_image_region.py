from typing import Union, List, Optional
import numpy as np
import dask.array as da
import zarr
import warnings

from .zoom import _zoom_numpy, _zoom_dask
from .ngff import _get_group_multiscales
from .downsampling import (
    SpatialFactor,
    level_scale_ratios_from_multiscales,
    normalize_spatial_factor,
    relative_level_factors,
)


def write_image_region(
    root: zarr.Group,
    mode: str,
    data: Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]],
    t: int | slice = slice(None),
    c: int | slice = slice(None),
    z: int | slice = slice(None),
    y: int | slice = slice(None),
    x: int | slice = slice(None),
    level: int = 0,
    group_name: Optional[str] = None,
    downscale_factor: SpatialFactor | None = None,
):
    """
    Internal function that writes image data (pyramid or single array) to a zarr group.

    If a single array is provided, the function automatically generates
    the complete pyramid by upsampling and downsampling from the given level.

    Parameters
    ----------
    root : zarr.Group
        The root Zarr group.
    mode : str
        Zarr open mode ("r+", "a", or "w"). Must allow writing.
    data : np.ndarray, da.Array, or list thereof
        Array(s) to write. If a list, each entry corresponds to one resolution level.
        If a single array, pyramid levels will be generated automatically.
    t, c, z, y, x : int or slice
        Indices/slices for each dimension.
    level : int
        Pyramid level of the provided `data` if `data` is a single array.
    group_name : str, optional
        Name of the subgroup inside the root. If None, writes to root image.
    """
    if mode not in ("r+", "a", "w"):
        raise PermissionError(
            f"Dataset opened in read-only mode ('{mode}'). "
            "Reopen with mode='r+' to allow modifications."
        )

    group = root if group_name is None else (root[group_name] if group_name in root else None)
    if group is None:
        available = list(root.group_keys())
        warnings.warn(
            f"Group '{group_name}' not found. Available groups: {available}",
            UserWarning,
        )
        return

    multiscales = _get_group_multiscales(group)
    if not multiscales:
        warnings.warn(
            f"No 'multiscales' attribute found in group '{group_name or '/'}'. "
            "Nothing to write.",
            UserWarning,
        )
        return

    multiscales = multiscales[0]
    datasets = multiscales.get("datasets", [])
    n_levels = len(datasets)

    if not 0 <= level < n_levels:
        raise ValueError(f"`level` must be in [0, {n_levels}), got {level}.")

    if downscale_factor is None:
        level_scale_ratios = level_scale_ratios_from_multiscales(
            multiscales,
            datasets,
            level,
        )

        if level_scale_ratios is None:
            level_scale_ratios = relative_level_factors(
                total_levels=n_levels,
                ref_level=level,
                downscale_factor=2,
            )
    else:
        level_scale_ratios = relative_level_factors(
            total_levels=n_levels,
            ref_level=level,
            downscale_factor=downscale_factor,
        )

    if n_levels == 0:
        warnings.warn(
            f"Group '{group_name or '/'}' contains no pyramid datasets.",
            UserWarning,
        )
        return

    if isinstance(data, (np.ndarray, da.Array)):
        data_list = _generate_pyramid(
            data,
            total_levels=n_levels,
            ref_level=level,
            level_scale_ratios=level_scale_ratios,
        )
    elif isinstance(data, list):
        data_list = data
        if len(data_list) != n_levels:
            warnings.warn(
                f"Provided pyramid has {len(data_list)} levels, "
                f"but dataset expects {n_levels}.",
                UserWarning,
            )
    else:
        raise TypeError("`data` must be a NumPy array, Dask array, or list of such.")

    for i, subdata in enumerate(data_list):
        scale_factor = level_scale_ratios[i]
        
        if i >= n_levels:
            break

        arr_path = datasets[i]["path"]
        if arr_path not in group:
            warnings.warn(
                f"Dataset path '{arr_path}' not found in group '{group_name or '/'}'.",
                UserWarning,
            )
            continue

        zarr_array = group[arr_path]
        
        index = _scale_index(
            (t, c, z, y, x),
            subdata.shape,
            scale_factor,
        )

        if isinstance(subdata, da.Array):
            subdata = subdata.compute()

        if subdata.ndim != 5:
            warnings.warn(
                f"Image write expects 5D data (tczyx), got shape {subdata.shape}. "
                f"Skipping level {i}.",
                UserWarning,
            )
            continue

        expected_shape = zarr_array[index].shape
        if subdata.shape != expected_shape:
            warnings.warn(
                f"Shape mismatch for level {i}: data={subdata.shape}, "
                f"expected={expected_shape}. Skipping level.",
                UserWarning,
            )
            continue

        zarr_array[index] = subdata

    store = getattr(root, "store", None)
    if store is not None and hasattr(store, "flush"):
        store.flush()

def _generate_pyramid(
    ref_data,
    total_levels: int,
    ref_level: int = 0,
    *,
    downscale_factor: SpatialFactor = 2,
    level_scale_ratios: list[tuple[float, float, float]] | None = None,
):
    """Generate a full pyramid from a given reference level.

    `level_scale_ratios[i]` is the spatial scale of level i relative to
    `ref_level`, in ZYX order.
    """
    if level_scale_ratios is None:
        level_scale_ratios = relative_level_factors(
            total_levels=total_levels,
            ref_level=ref_level,
            downscale_factor=downscale_factor,
        )

    if len(level_scale_ratios) != total_levels:
        raise ValueError(
            "`level_scale_ratios` must contain one entry per pyramid level."
        )

    zoom_fn = _zoom_dask if isinstance(ref_data, da.Array) else _zoom_numpy

    pyramid = [None] * total_levels
    pyramid[ref_level] = ref_data

    # Generate finer levels.
    for i in range(ref_level - 1, -1, -1):
        scale = tuple(
            level_scale_ratios[i + 1][axis] / level_scale_ratios[i][axis]
            for axis in range(3)
        )
        pyramid[i] = zoom_fn(pyramid[i + 1], scale=scale)

    # Generate coarser levels.
    for i in range(ref_level + 1, total_levels):
        scale = tuple(
            level_scale_ratios[i - 1][axis] / level_scale_ratios[i][axis]
            for axis in range(3)
        )
        pyramid[i] = zoom_fn(pyramid[i - 1], scale=scale)

    return pyramid


def _scale_index(index_tuple, shape, scale_factor: SpatialFactor):
    """Scale only spatial indices for image arrays shaped T, C, Z, Y, X."""
    zyx_factors = normalize_spatial_factor(
        scale_factor,
        name="scale_factor",
        allow_float=True,
    )

    t, c, z, y, x = index_tuple

    def scale_spatial(sel, size, factor):
        if isinstance(sel, int):
            return int(sel / factor)

        start = None if sel.start is None else int(sel.start / factor)
        stop = size if start is None else start + size

        return slice(start, stop, None)

    return (
        t,
        c,
        scale_spatial(z, shape[2], zyx_factors[0]),
        scale_spatial(y, shape[3], zyx_factors[1]),
        scale_spatial(x, shape[4], zyx_factors[2]),
    )
