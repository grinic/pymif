from typing import Union, List, Optional
import numpy as np
import dask.array as da
import zarr
import warnings


def write_region(
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
    Internal function that writes data (pyramid or single array) to a zarr group.

    If a single array is provided, the function will automatically generate
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
        Slices for each dimension.
    level : int
        The pyramid level to write to (if `data` is a single array).
    group_name : str, optional
        Name of the group inside the root (e.g. "image" or "labels").
    """
    # Check mode
    if mode not in ("r+", "a", "w"):
        raise PermissionError(
            f"Dataset opened in read-only mode ('{mode}'). "
            "Reopen with mode='r+' to allow modifications."
        )

    # Select target group
    group = root if group_name is None else root.get(group_name)
    print(group)
    if group is None:
        available = list(root.group_keys())
        warnings.warn(
            f"Group '{group_name}' not found. Available groups: {available}",
            UserWarning,
        )
        return

    # Get multiscales metadata
    multiscales = group.attrs.get("multiscales")
    if not multiscales:
        warnings.warn(
            f"No 'multiscales' attribute found in group '{group_name or '/'}'. "
            "Nothing to write.",
            UserWarning,
        )
        return
    multiscales = multiscales[0]
    n_levels = len(multiscales["datasets"])

    # Handle input data
    if isinstance(data, (np.ndarray, da.Array)):
        # Generate full pyramid around the provided level
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

    # Write data
    for i, subdata in enumerate(data_list):
        arr_path = multiscales["datasets"][i]["path"]
        if arr_path not in group:
            warnings.warn(f"Dataset path '{arr_path}' not found in group '{group_name or '/'}'.")
            continue

        zarr_array = group[arr_path]

        # Dask → NumPy
        if isinstance(subdata, da.Array):
            subdata = subdata.compute()

        # Scale slices for this level, 
        # take into account that if subdata has 4 dimensions, it's a label, drop c
        scale_factor = 2 ** (i - level)
        index = _scale_index((t, c, z, y, x), subdata.shape, scale_factor)

        print(subdata.shape, index)

        # Shape consistency
        # print (index, subdata.shape, zarr_array[index].shape)
        if subdata.shape != zarr_array[index].shape:
            warnings.warn(
                f"Shape mismatch for level {i}: data={subdata.shape}, "
                f"expected={zarr_array.shape}. Skipping level.",
                UserWarning,
            )
            continue

        # Update the data
        zarr_array[index] = subdata

    # Flush to disk if possible
    store = getattr(root, "store", None)
    if store and hasattr(store, "flush"):
        store.flush()
    
def _generate_pyramid(
    ref_data: Union[np.ndarray, da.Array],
    total_levels: int,
    ref_level: int = 0,
) -> List[Union[np.ndarray, da.Array]]:
    """
    Generate a full pyramid (both upscaling and downscaling) from a given level.

    Parameters
    ----------
    ref_data : np.ndarray or dask.array.Array
        The reference data (e.g. from level 2).
    total_levels : int
        Total number of levels in the pyramid.
    ref_level : int
        Level index of the provided `ref_data`.

    Returns
    -------
    list of arrays
        One array per pyramid level, from level 0 (highest res) to N-1 (lowest res).
    """
    if isinstance(ref_data, da.Array):
        base_type = da
        zoom_fn = _zoom_dask
    else:
        base_type = np
        zoom_fn = _zoom_numpy

    pyramid = [None] * total_levels
    pyramid[ref_level] = ref_data

    # Upscale (toward level 0)
    for i in range(ref_level - 1, -1, -1):
        pyramid[i] = zoom_fn(pyramid[i + 1], scale=2.0)

    # Downscale (toward lower resolutions)
    for i in range(ref_level + 1, total_levels):
        pyramid[i] = zoom_fn(pyramid[i - 1], scale=0.5)

    return pyramid

def _scale_index(index_tuple, shape, scale_factor: float):
    """Scale slices or ints for down/up-sampled levels."""
    if len(index_tuple) == 5:
        t, c, z, y, x = index_tuple
    else:
        t, z, y, x = index_tuple

    def scale_slice(s, size):
        start = None if s.start is None else int(s.start / scale_factor)
        stop = start + size

        return slice(start, stop, None)

    # Typically only y, x (and possibly z) are scaled
    if len(shape)==5:
        return (
            t,
            c,
            scale_slice(z, shape[2]),
            scale_slice(y, shape[3]),
            scale_slice(x, shape[4]),
        )
    else:
        return (
            t,
            scale_slice(z, shape[1]),
            scale_slice(y, shape[2]),
            scale_slice(x, shape[3]),
        )

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
    import dask_image.ndinterp as ndinterp
    factors = [1] * (arr.ndim - 3) + [scale, scale, scale]
    for i, f in enumerate(factors):
        if (arr.shape[i]*f) < 1:
            factors[i] = 1
    return ndinterp.zoom(arr, zoom=factors, order=0)
