import dask.array as da
from numcodecs import Blosc, GZip
from typing import List, Dict, Any
from ome_zarr.writer import write_multiscale
import zarr, os, shutil
from pathlib import Path

DEFAULT_COLORS = [
    "FF0000",  # Red
    "00FF00",  # Green
    "0000FF",  # Blue
    "FFFF00",  # Yellow
    "FF00FF",  # Magenta
    "00FFFF",  # Cyan
    "FFFFFF",  # White
    "808080",  # Gray
]

def add_label(
    root: zarr.Group,
    mode: str,
    label_levels: List[da.Array],
    label_name: str,
    metadata: Dict[str, Any],
    compressor=None,
    compressor_level=3,
    parallelize=False,
):
    """
    Write image data and metadata to the specified path.

    Parameters
    ----------
        path : str
            Destination path for the output data.
        label_levels : List[da.Array]
            List of Dask arrays representing image data.
        label_name : str
            Name of the objects labeled
        metadata : Dict[str, Any]
            Dictionary containing metadata information.
    """
    if mode not in ("r+", "a", "w"):
        raise PermissionError(
            f"Dataset opened in read-only mode ('{mode}'). "
            "Reopen with mode='r+' to allow modifications."
        )

    expected_ndim = 4
    label_layers = len(label_levels)
    expected_layers = len(metadata["size"])
    if label_layers != expected_layers:
        raise ValueError(
            f"Label pyramid has {label_layers} levels, expected {expected_layers}."
        )
        
    for i, level in enumerate(label_levels):
        expected_shape = (
                            metadata["size"][i][0],
                            metadata["size"][i][2],
                            metadata["size"][i][3],
                            metadata["size"][i][4],
        )
        if level.ndim != expected_ndim:
            raise ValueError(
                f"Label pyramid level {i} has {level.ndim} dimensions, expected {expected_ndim} (tzyx)."
            )
        if level.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch at level {i}. Label shape: {level.shape}, expected shape: {expected_shape}"
            )
    
    
    if isinstance(compressor, str):
        if compressor.lower() == "blosc":
            compressor = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.BITSHUFFLE)
        if compressor.lower() == "gzip":
            compressor = GZip(level=compressor_level)
        
    scales = metadata["scales"]  # [[1,0.173,0.173], [2,0.346,0.346], ...]
    
    coordinate_transformations = [
        [
            {
                "type": "scale", 
                "scale": [metadata["time_increment"]] + list(scale),
            }
        ]
        for scale in scales
    ]

    labels_grp = root.require_group("labels")
    
    # Remove the existing label group if it exists (clean reset)
    if label_name in labels_grp:
        del labels_grp[label_name]
    
    label_grp = labels_grp.create_group(label_name)

    label_grp.attrs["image-label"] = {
            "source": {"image": "../../"}
    }
    write_multiscale(
        pyramid=label_levels,
        group=label_grp,
        axes="tzyx",
        coordinate_transformations=coordinate_transformations,
        # storage_options={"compressor": compressor},
    )

    # Check if label already listed
    # Update labels list at the root (just names, not paths)
    labels_attr = root.attrs.get("labels", [])
    if not isinstance(labels_attr, list):
        labels_attr = []
    # Check if label already listed
    if not any(lbl == f"labels/{label_name}" for lbl in labels_attr):
        labels_attr.append(f"labels/{label_name}")
    
    # Update root attributes with full label list
    root.attrs["labels"] = labels_attr
