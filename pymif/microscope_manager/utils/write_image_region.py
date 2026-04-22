from typing import Union, List, Optional
import numpy as np
import dask.array as da
import zarr
import warnings

from .zoom import _zoom_numpy, _zoom_dask
from .ngff import _get_group_multiscales


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

    if n_levels == 0:
        warnings.warn(
            f"Group '{group_name or '/'}' contains no pyramid datasets.",
            UserWarning,
        )
        return

    if isinstance(data, (np.ndarray, da.Array)):
        data_list = _generate_pyramid(data, total_levels=n_levels, ref_level=level)
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

        if isinstance(subdata, da.Array):
            subdata = subdata.compute()

        if subdata.ndim != 5:
            warnings.warn(
                f"Image write expects 5D data (tczyx), got shape {subdata.shape}. "
                f"Skipping level {i}.",
                UserWarning,
            )
            continue

        scale_factor = 2 ** (i - level)
        index = _scale_index((t, c, z, y, x), subdata.shape, scale_factor)

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
    ref_data: Union[np.ndarray, da.Array],
    total_levels: int,
    ref_level: int = 0,
) -> List[Union[np.ndarray, da.Array]]:
    """
    Generate a full pyramid from a given reference level.
    """
    zoom_fn = _zoom_dask if isinstance(ref_data, da.Array) else _zoom_numpy

    pyramid = [None] * total_levels
    pyramid[ref_level] = ref_data

    for i in range(ref_level - 1, -1, -1):
        pyramid[i] = zoom_fn(pyramid[i + 1], scale=2.0)

    for i in range(ref_level + 1, total_levels):
        pyramid[i] = zoom_fn(pyramid[i - 1], scale=0.5)

    return pyramid


def _scale_index(index_tuple, shape, scale_factor: float):
    """
    Scale only spatial indices (z, y, x) for a target pyramid level.

    t and c are never scaled.
    `shape` is the shape of the data being written at that target level.
    """
    t, c, z, y, x = index_tuple

    def scale_spatial(sel, size):
        if isinstance(sel, int):
            return int(sel / scale_factor)

        start = None if sel.start is None else int(sel.start / scale_factor)
        stop = start + size if start is not None else size
        return slice(start, stop, None)

    return (
        t,
        c,
        scale_spatial(z, shape[2]),
        scale_spatial(y, shape[3]),
        scale_spatial(x, shape[4]),
    )