import dask.array as da
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from numcodecs import Blosc, GZip
from typing import List, Dict, Any
from ome_zarr.writer import write_multiscale
import zarr
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

def to_zarr(
    path: str,
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    compressor: Any =None,
    compressor_level: int =3,
    overwrite: bool =True,
    parallelize: bool =False,
):
    """
    Write image data and metadata to the specified path.

    Parameters
    ----------
        root : str
            Destination root for the output data.
        data_levels : List[da.Array]
            List of Dask arrays representing image data.
        metadata : Dict[str, Any]
            Dictionary containing metadata information.
        compressor : Any
            Type of compression used (default: None).
        compressor_level : int
            Compression level used (if compression is not None).
        overwrite : bool
            whether to overwrite existing data at Destination path (default: True).
        parallelize : bool
            whether to use dask distribute to parallelize (default: False).
    """
    print("Start writing dataset.")

    root = zarr.open(zarr.NestedDirectoryStore(path), mode="w")
    
    if isinstance(compressor, str):
        if compressor.lower() == "blosc":
            compressor = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.BITSHUFFLE)
        if compressor.lower() == "gzip":
            compressor = GZip(level=compressor_level)
        
    store_path = Path(root.store.path)
    if store_path.exists() and overwrite:
        import shutil
        shutil.rmtree(store_path)

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
    
    print("Writing pyramid.")
    write_multiscale(
        pyramid=data_levels,
        group=root,
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

    root.attrs["multiscales"] = [{
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

    root.attrs["omero"] = {
        "channels": channels,
        "rdefs": {"model": "color"}
    }

