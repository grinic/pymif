import dask.array as da
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from numcodecs import Blosc, GZip
from typing import List, Dict, Any, Optional
from ome_zarr.format import CurrentFormat
from ome_zarr.writer import write_multiscale
import zarr, os, shutil
from pathlib import Path
import numpy as np

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
    path: str,
    label_levels: List[da.Array],
    metadata: Dict[str, Any],
    label_name: Optional[str] = "new_label",
    compressor=None,
    compressor_level=3,
    parallelize=False,
):
    """
    Write image data and metadata to the specified path.

    Args:
        path (str): Destination path for the output data.
        data (List[da.Array]): List of Dask arrays representing image data.
        metadata (Dict[str, Any]): Dictionary containing metadata information.
    """
    
    if isinstance(compressor, str):
        if compressor.lower() == "blosc":
            compressor = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.BITSHUFFLE)
        if compressor.lower() == "gzip":
            compressor = GZip(level=compressor_level)
        
    store_path = Path(path)

    store = zarr.NestedDirectoryStore(str(store_path))
    root_group = zarr.group(store=store)

    scales = metadata["scales"]  # [[1,0.173,0.173], [2,0.346,0.346], ...]
    axes_labels = metadata["axes"]
    
    coordinate_transformations = [
        [
            {
                "type": "scale", 
                "scale": [metadata["time_increment"]] + list(scale),
            }
        ]
        for scale in scales
    ]

    labels_grp = root_group.require_group("labels")
    
    label_list = [label_name]
    
    # Read current labels metadata once, outside loop
    labels_attr = root_group.attrs.get("labels", [])
    if not isinstance(labels_attr, list):
        labels_attr = []
    
    # Remove the existing label group if it exists (clean reset)
    full_label_path = os.path.join(path, f"labels/{label_name}")
    if os.path.exists(full_label_path):
        shutil.rmtree(full_label_path)
    
    label_grp = labels_grp.create_group(label_name)

    label_grp.attrs["image-label"] = {
            # "colors": get_zarr_labels_color_metadata(nuc_preds),
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
    if not any(lbl == label_name for lbl in labels_attr):
        labels_attr.append(f"labels/{label_name}")
        
    # Update root attributes with full label list
    root_group.attrs["labels"] = labels_attr










    axes_map = {
        "t": "time",
        "z": "space",
        "y": "space",
        "x": "space",
    }
    units = [metadata["time_increment_unit"]] + list( metadata["units"] )
    def normalize_unit(unit: str) -> str:
        # Common aliases to normalize
        if not unit:
            return unit 
        aliases = {
            "um": f"micrometer",
            "μm": f"micrometer",  # Greek mu
            "\u00b5m": f"micrometer",  # Micro sign
            "micron": f"micrometer",
            "microns": f"micrometer",
        }
        return aliases.get(unit.strip(), unit.strip())
    
    axes = [
            {
                "name": ax, 
                "type": axes_map.get(ax, "unknown"),
                "unit": normalize_unit(units[i])
            } for i, ax in enumerate(axes_labels)
        ]

    # Write multiscale array in root
    if parallelize:
        client = Client()
        print("Dask dashboard:", client.dashboard_link)
        ProgressBar().register()
    
    write_multiscale(
        pyramid=data_levels,
        group=root_group,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options={"compressor": compressor},
    )

    # OMERO metadata
    C = data_levels[0].shape[axes_labels.index("c")]
    ch_names = metadata.get("channel_names", [f"channel_{i}" for i in range(C)])
    ch_colors = metadata.get("channel_colors", ["FFFFFF"] * C)
    
    def _normalize_color(color):
        """Ensure color is a 6-digit hex string."""
        if isinstance(color, int):
            return f"{color & 0xFFFFFF:06X}"  # mask to 24-bit and format
        if isinstance(color, str):
            color = color.lstrip("#-")
            if len(color) == 6:
                return color.upper()
        return "FFFFFF"  # default fallback

    channels = [{
        "label": ch_names[i],
        "color": _normalize_color(ch_colors[i]),
        "window": {
            "start": 0,
            "end": 1500,
            "min": 0,
            "max": 65535
        },
        "active": True,
        "inverted": False,
        "coefficient": 1.0,
        "family": "linear",
    } for i in range(C)]

    root_group.attrs["multiscales"] = [{
        # "version": CurrentFormat.version,
        "name": metadata.get("name", "OME-Zarr image"),
        "datasets": [
            {
                "path": str(i),
                "coordinateTransformations": coordinate_transformations[i],
            }
            for i in range(len(data_levels))
        ],
        "axes": axes,
        "type": "image",
    }]

    root_group.attrs["omero"] = {
        "channels": channels,
        "rdefs": {"model": "color"}
    }

