import dask.array as da
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from numcodecs import Blosc, GZip
from typing import List, Dict, Any
from ome_zarr.format import CurrentFormat
from ome_zarr.writer import write_multiscale
import zarr
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

def write(
    path: str,
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    compressor=None,
    compressor_level=3,
    overwrite=True,
    parallelize=False,
):
    """
    Write image data and metadata to the specified path.

    Args:
        path (str): Destination path for the output data.
        data (List[da.Array]): List of Dask arrays representing image data.
        metadata (Dict[str, Any]): Dictionary containing metadata information.
    """
    
    print(parallelize)

    if compressor is "Blosc":
        compressor = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.BITSHUFFLE)
    if compressor.lower() is "gzip":
        compressor = GZip(level=compressor_level)
        
    store_path = Path(path)
    if store_path.exists() and overwrite:
        import shutil
        shutil.rmtree(store_path)

    store = zarr.NestedDirectoryStore(str(store_path))
    root_group = zarr.group(store=store, overwrite=True)

    scales = metadata["scales"]  # [[1,0.173,0.173], [2,0.346,0.346], ...]
    axes_labels = metadata["axes"]
    
    coordinate_transformations = [
        [
            {
                "type": "scale", 
                "scale": [metadata["time_increment"]] + [1] + list(scale),
            }
        ]
        for scale in scales
    ]

    axes_map = {
        "t": "time",
        "c": "channel",
        "z": "space",
        "y": "space",
        "x": "space",
    }
    units = [metadata["time_increment_unit"], ""] + list( metadata["units"] )
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

