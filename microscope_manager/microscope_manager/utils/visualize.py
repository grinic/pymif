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
    viewer: Optional[napari.Viewer] = None,
) -> napari.Viewer:
    """Visualize a pyramidal image dataset using napari.
    
    Parameters:
        data_levels: list of Dask arrays (one per resolution level).
        metadata: dictionary containing scales, channel info, etc.
        viewer: optional napari Viewer to add the image to.
    
    Returns:
        napari.Viewer: the viewer with the image added.
    """
    if viewer is None:
        viewer = napari.Viewer()
        
    size = metadata['size'][0]

    channel_axis = 1 if 'channel_names' in metadata and size[1] > 1 else None
    num_channels = size[1]

    if "channel_colors" in metadata:
        colormaps = [
            _ome_int_to_rgb_tuple(c) for c in metadata["channel_colors"]
        ]
    else:
        colormaps = ["gray"] * num_channels

    viewer.add_image(
        data_levels,
        name="Pyramidal Image",
        scale=metadata.get("scales", [(1, 1, 1)])[0],
        channel_axis=channel_axis,
        colormap=colormaps,
        metadata=metadata,
        contrast_limits=[0,2000],
        multiscale=True
    )

    return viewer

