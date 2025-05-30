import napari
import dask.array as da
from typing import List, Dict, Any, Optional

def _ome_int_to_rgb_tuple(color_int: int) -> tuple:
    """Convert OME int color to RGB float tuple for Napari."""
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return (r / 255.0, g / 255.0, b / 255.0)

def visualize(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    start_level: Optional[int] = 0,
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

    if viewer is None:
        viewer = napari.Viewer()

    # Reduce pyramid to selected levels
    pyramid = data_levels[start_level:]

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
    if "channel_colors" in metadata:
        colormaps = [
            _ome_int_to_rgb_tuple(c) for c in metadata["channel_colors"]
        ]
    else:
        colormaps = ["gray"] * num_channels

    viewer.add_image(
        pyramid,
        name="Pyramidal Image",
        scale=metadata.get("scales", [(1, 1, 1)])[0],
        channel_axis=channel_axis,
        colormap=colormaps,
        metadata=metadata,
        contrast_limits=[0,2000],
        multiscale=True
    )

    return viewer

