from typing import Dict, Any
import zarr
# from ome_zarr.format import CurrentFormat

def create_empty_dataset(
    root: zarr.Group,
    metadata: Dict[str, Any],
):
    """
    Create an empty OME-Zarr dataset using provided metadata (lazy initialization).
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty dataset.")

    axes_labels = metadata["axes"]
    sizes = metadata["size"]       # list of shapes per level
    chunks = metadata["chunksize"] # list of chunk shapes
    dtype = metadata.get("dtype", "uint16")
    scales = metadata.get("scales", [])
    axes = metadata.get("axes", [])
    units = metadata.get("units", [])
    time_increment = metadata.get("time_increment", 1.0)
    
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
    
    # Create an NGFF multiscale structure manually, without writing data
    datasets = []
    for i, (shape, chunk) in enumerate(zip(sizes, chunks)):
        arr = root.create_dataset(
            name=str(i),
            shape=shape,
            chunks=chunk,
            dtype=dtype,
            # compressor=None,
            # write_empty_chunks=False,  # ✅ prevents physical chunk creation
        )
        datasets.append({"path": str(i)})

    # OMERO metadata
    C = sizes[0][axes_labels.index("c")]
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


    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": metadata.get("name", "OME-Zarr image"),
            "datasets": [
                {
                    "path": str(i),
                    "coordinateTransformations": coordinate_transformations[i],
                }
                for i in range(len(sizes))
            ],
            "axes": axes,
            "type": "image",
        }],
        "omero": {
            "channels": channels,
            "rdefs": {"model": "color"}
        }
    }


    print(f"[INFO] Created empty OME-Zarr dataset at {root.store}.")

