from typing import Dict, Any
from ome_zarr.writer import write_multiscale, add_metadata
import zarr

def create_empty_group(
    root: zarr.Group,
    group_name: str,
    metadata: Dict[str, Any],
    is_label: bool = False,
):
    """
    Create an empty OME-Zarr image or label group using provided metadata.

    Parameters
    ----------
    root : zarr.Group
        root group.
    group_name : str
        Name of the new subgroup (e.g. "processed" or "labels/nuclei").
    metadata : Dict[str, Any]
        Metadata describing the dataset structure.
    is_label : bool, optional
        If True, create a label group inside 'labels/' instead of a new image group.
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty group.")

    # Determine target group
    if is_label:
        labels_grp = root.require_group("labels")
        if group_name in labels_grp:
            del labels_grp[group_name]
        grp = labels_grp.create_group(group_name)
    else:
        if group_name in root:
            del root[group_name]
        grp = root.create_group(group_name)

    sizes = metadata["size"]
    chunks = metadata["chunksize"]
    dtype = metadata.get("dtype", "uint16")
    scales = metadata.get("scales", [])
    time_increment = metadata.get("time_increment", 1.0)
    time_increment_unit = metadata.get("time_increment_unit", "")
    axes_labels = metadata["axes"]
    units = metadata.get("units", [])

    # Build coordinate transformations
    coordinate_transformations = [
        [
            {
                "type": "scale",
                "scale": [time_increment] + ([1] if "c" in axes_labels and not is_label else []) + list(scale),
            }
        ]
        for scale in scales
    ]

    # Axis metadata
    axes_map = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    units = [time_increment_unit] + ([] if is_label else [""]) + list(units)

    def normalize_unit(unit: str) -> str:
        aliases = {
            "um": "micrometer",
            "μm": "micrometer",
            "\u00b5m": "micrometer",
            "micron": "micrometer",
            "microns": "micrometer",
        }
        return aliases.get(unit.strip(), unit.strip()) if unit else unit

    axes = [
        {
            "name": ax,
            "type": axes_map.get(ax, "unknown"),
            "unit": normalize_unit(units[i]) if i < len(units) else "",
        }
        for i, ax in enumerate(axes_labels)
    ]

    # Create pyramid datasets lazily
    for i, (shape, chunk) in enumerate(zip(sizes, chunks)):
        grp.create_dataset(
            name=str(i),
            shape=shape,
            chunks=chunk,
            dtype=dtype,
            # compressor=None,
            # write_empty_chunks=False,  # defer physical chunk creation
        )

    # Add multiscale metadata
    multiscale_entry = {
        "version": "0.5",
        "name": group_name,
        "datasets": [{"path": str(i), "coordinateTransformations": coordinate_transformations[i]} for i in range(len(sizes))],
        "axes": axes,
    }

    grp.attrs["ome"]= {"multiscales": [multiscale_entry]}

    # Register in root so Napari can see it
    if is_label:
        grp.attrs["image-label"] = {"source": {"image": "../../"}}  # label points to root image
        labels_attr = root.attrs.get("labels", [])
        if not isinstance(labels_attr, list):
            labels_attr = []
        label_path = f"labels/{group_name}"
        if label_path not in labels_attr:
            labels_attr.append(label_path)
            root.attrs["labels"] = labels_attr
    else:
        # For images, append to root multiscales if not already present
        root_multiscales = root.attrs.get("ome").get("multiscales", [])
        root_multiscales.append({
            "name": group_name,
            "path": grp.path,
            "datasets": [{"path": str(i)} for i in range(len(sizes))],
            "axes": axes,
            "type": "image",
        })
        root.attrs.get("ome")["multiscales"] = root_multiscales

        # Optional OMERO metadata for images
        C = sizes[0][axes_labels.index("c")] if "c" in axes_labels else 1
        ch_names = metadata.get("channel_names", [f"channel_{i}" for i in range(C)])
        ch_colors = metadata.get("channel_colors", ["FFFFFF"] * C)
        def _normalize_color(color):
            if isinstance(color, int):
                return f"{color & 0xFFFFFF:06X}"
            if isinstance(color, str):
                color = color.lstrip("#-")
                if len(color) == 6:
                    return color.upper()
            return "FFFFFF"
        channels = [{
            "label": ch_names[i], 
            "color": _normalize_color(ch_colors[i]), 
            "window": {
                "start":0,
                "end":1500,
                "min":0,
                "max":65535
            },
            "active":True,
            # "inverted":False,
            # "coefficient":1.0,
            # "family":"linear"
        } for i in range(C) ]
        
        add_metadata(
            grp,
            {"omero":{
                    "channels": channels,
                    # "rdefs": {"model": "color"}
                }
            }
        )

        grp.attrs["image-source"] = {"source":{"image":"../"}}

    print(f"[INFO] Created empty {'label' if is_label else 'image'} group '{grp.path}' in store '{root.store}'")
    return grp
