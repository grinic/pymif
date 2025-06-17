import napari
import dask.array as da
from typing import List, Dict, Any, Optional, Union, Tuple

def _parse_color(color: Union[int, str]) -> tuple:
    """Convert OME int or hex string color to RGB float tuple for Napari."""
    if isinstance(color, int):
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
    elif isinstance(color, str):
        s = color.strip()
        if s.startswith("#"):
            if s.startswith("#-"):
                s = s[2:]
            else:
                s = s[1:]
        if s.lower().startswith("0x"):
            s = s[2:]

        if len(s) == 8:
            s = s[2:]  # drop AA from AARRGGBB

        if len(s) != 6:
            raise ValueError(f"Invalid hex color string: {color!r} (expected 6 or 8 hex digits)")

        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)

    else:
        raise TypeError(f"Unsupported color type: {type(color)}")
    return (r / 255.0, g / 255.0, b / 255.0)

def visualize(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    start_level: Optional[int] = 0,
    stop_level: Optional[int] = -1,
    in_memory: Optional[bool] = False,
    viewer: Optional[napari.Viewer] = None,
    ) -> napari.Viewer:
    """
    Visualize a multiscale dataset with Napari.

    Args:
        data_levels: list of Dask arrays, one per resolution level.
        metadata: dict with NGFF-compatible metadata.
        viewer: optional existing Napari viewer.
        start_level: index of resolution level to use as the highest-resolution (default: 0).
        in_memory: if True, convert Dask arrays to NumPy for interactivity (default: False).

    Returns:
        napari.Viewer: viewer instance with the image loaded.
    """
    # Check that start_level is valid
    if not 0 <= start_level < len(data_levels):
        raise ValueError(f"start_level={start_level} is out of bounds for available {len(data_levels)} levels.")
    if (stop_level>0) & (not stop_level < len(data_levels)):
        raise ValueError(f"stop_level={stop_level} is out of bounds for available {len(data_levels)} levels.")
    if (stop_level>0) & (not start_level < stop_level):
        raise ValueError(f"start_level={start_level} must be lower than stop_level={stop_level}.")

    if viewer is None:
        viewer = napari.Viewer()

    # Subselect pyramid levels
    pyramid = data_levels[start_level:] if stop_level == -1 else data_levels[start_level:stop_level]

    # If requested, convert Dask arrays to NumPy for interactivity
    if in_memory:
        try:
            pyramid = [p.compute() for p in pyramid]
        except Exception as e:
            raise RuntimeError(f"Failed to load data into memory: {e}")    
            
    size = metadata.get("size", [pyramid[0].shape])
    num_channels = size[0][1] if len(size[0]) >= 2 else 1
    channel_axis = 1 if num_channels > 1 else None
    
    # Handle channel colors
    channel_colors = metadata.get("channel_colors", [])
    if channel_colors:
        colormaps = [_parse_color(c) for c in channel_colors]
    else:
        colormaps = ["gray"] * num_channels

    max_val = da.max(pyramid[-1])
    clim_max = max(1, int(2 * max_val))
    viewer.add_image(
        pyramid,
        name=metadata.get("channel_names", [f"ch_{i}" for i in range(num_channels)]),
        scale=metadata.get("scales", [(1, 1, 1)])[start_level],
        channel_axis=channel_axis,
        colormap=colormaps,
        metadata=metadata,
        contrast_limits=[0, clim_max],
        multiscale=True
    )

    # --- Add label layers if present
    label_metadata = metadata.get("labels_metadata", {})
    for label_name, label_info in label_metadata.items():
        label_pyramid = label_info.get("data")  # list of Dask arrays
        label_pyramid = label_pyramid[start_level:] if stop_level == -1 else label_pyramid[start_level:stop_level]

        label_scale = label_info.get("scale", metadata.get("scales", [(1, 1, 1)])[start_level])
        # label_color = label_info.get("color", "magenta")
        label_opacity = label_info.get("opacity", 0.5)

        if in_memory:
            try:
                label_pyramid = [l.compute() for l in label_pyramid]
            except Exception as e:
                raise RuntimeError(f"Failed to load labels into memory: {e}")    

        viewer.add_labels(
            label_pyramid,
            name=label_name,
            scale=label_scale,
            multiscale=True,
            # colormap=label_color,
            opacity=label_opacity,
        )

    return viewer

