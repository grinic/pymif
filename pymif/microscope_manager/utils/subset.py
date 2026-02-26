import numpy as np
import dask.array as da
from typing import Optional, Union, Sequence

def subset_dask_array(
    arr: da.Array,
    T: Optional[Union[slice, Sequence[int]]] = None,
    C: Optional[Union[slice, Sequence[int]]] = None,
    Z: Optional[Union[slice, Sequence[int]]] = None,
    Y: Optional[Union[slice, Sequence[int]]] = None,
    X: Optional[Union[slice, Sequence[int]]] = None,
) -> da.Array:
    """Subset a 5D Dask array with optional indexing for each axis."""
    axes = [T, C, Z, Y, X]
    for axis, index in enumerate(axes):
        if index is not None:
            index = np.array(index)
            if index.ndim != 1:
                raise NotImplementedError("Only 1D fancy indexing is supported.")
            arr = arr[tuple(slice(None) if i != axis else index for i in range(arr.ndim))]
    return arr

def validate_uniform_spacing(indices: Sequence[int], name: str):
    """Ensure spacing between indices is uniform."""
    if len(indices) < 2:
        return 1  # No spacing to validate for 0 or 1 index
    diffs = np.diff(indices)
    if not np.all(diffs == diffs[0]):
        raise ValueError(f"Non-uniform spacing in {name} axis: {indices}")
    return diffs[0]

def subset_metadata(metadata: dict, 
                    T: Optional[Sequence[int]] = None,
                    C: Optional[Sequence[int]] = None,
                    Z: Optional[Sequence[int]] = None,
                    Y: Optional[Sequence[int]] = None,
                    X: Optional[Sequence[int]] = None) -> dict:
    """Update metadata after subsetting."""
    new_metadata = metadata.copy()
    shape = list(metadata["size"][0])  # start from the first level always

    axis_map = {"t": T, "c": C, "z": Z, "y": Y, "x": X}
    axes = metadata["axes"]

    new_size = list(shape)
    for i, ax in enumerate(axes):
        index = axis_map[ax]
        if index is not None:
            new_size[i] = len(index)
    new_metadata["size"] = [tuple(new_size)]

    # Adjust time increment if T is subset
    if T is not None:
        t_gap = validate_uniform_spacing(T, "T")
        new_metadata["time_increment"] = t_gap * metadata["time_increment"]
        
    # Adjust scales for Z, Y, X
    new_scales = list(metadata["scales"][0])
    for i, ax in enumerate("zyx"):
        idx = axis_map[ax]
        if idx is not None:
            spacing = validate_uniform_spacing(idx, ax.upper())
            new_scales[i] *= spacing
    new_metadata["scales"] = [tuple(new_scales)]

    # Subset channel names/colors
    if C is not None:
        new_metadata["channel_names"] = [metadata["channel_names"][i] for i in C]
        new_metadata["channel_colors"] = [metadata["channel_colors"][i] for i in C]

    return new_metadata
