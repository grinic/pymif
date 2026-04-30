from __future__ import annotations

import warnings
from typing import Any, Dict, List, TYPE_CHECKING, Union

import dask.array as da

from .axes import normalize_axes, spatial_axes_in_order

if TYPE_CHECKING:
    import napari


def _parse_color(color: Union[int, str]) -> tuple[float, float, float]:
    """Convert OME int or hex string color to RGB float tuple for Napari."""
    if isinstance(color, int):
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
    elif isinstance(color, str):
        s = color.strip()
        if s.startswith("#"):
            s = s[1:]
        if s.lower().startswith("0x"):
            s = s[2:]

        if len(s) == 8:
            s = s[2:]  # drop AA from AARRGGBB

        if len(s) != 6:
            raise ValueError(
                f"Invalid hex color string: {color!r} (expected 6 or 8 hex digits)"
            )

        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    else:
        raise TypeError(f"Unsupported color type: {type(color)}")

    return (r / 255.0, g / 255.0, b / 255.0)

def _axis_scale(metadata: Dict[str, Any], axes: tuple[str, ...], level: int, *, drop_channel: bool) -> tuple[float, ...]:
    spatial_axes = spatial_axes_in_order(axes)
    spatial_scale = metadata.get("scales", [tuple(1.0 for _ in spatial_axes)])[level]
    spatial_map = dict(zip(spatial_axes, spatial_scale))
    scale = []
    for ax in axes:
        if drop_channel and ax == "c":
            continue
        if ax == "t":
            scale.append(float(metadata.get("time_increment") or 1.0))
        elif ax in spatial_map:
            scale.append(float(spatial_map[ax]))
        else:
            scale.append(1.0)
    return tuple(scale)


def _set_axis_labels(viewer, axes: tuple[str, ...], *, drop_channel: bool) -> None:
    labels = tuple(ax.upper() for ax in axes if not (drop_channel and ax == "c"))
    try:
        if len(labels) == len(viewer.dims.axis_labels):
            viewer.dims.axis_labels = labels
    except Exception:
        pass

def visualize(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    start_level: int = 0,
    stop_level: int = -1,
    in_memory: bool = False,
    viewer: "napari.Viewer | None" = None,
) -> "napari.Viewer | None":
    """Visualize an axis-aware multiscale dataset with napari.

    Datasets without a ``t`` axis are passed to napari without an artificial time
    dimension, so napari will not expose an active T slider.  Datasets without a
    ``c`` axis are displayed as a single image/label layer rather than channel
    layers.
    """
    try:
        import napari
    except ImportError:
        warnings.warn(
            "napari is not installed. Install with `pip install pymif[napari]` "
            "to use visualization.",
            stacklevel=2,
        )
        return None

    if not data_levels:
        raise ValueError("No data levels supplied for visualization.")
    if not 0 <= start_level < len(data_levels):
        raise ValueError(f"start_level={start_level} is out of bounds for {len(data_levels)} levels.")
    if stop_level > 0 and stop_level > len(data_levels):
        raise ValueError(f"stop_level={stop_level} is out of bounds for {len(data_levels)} levels.")
    if stop_level > 0 and start_level >= stop_level:
        raise ValueError(f"start_level={start_level} must be lower than stop_level={stop_level}.")

    if viewer is None:
        viewer = napari.Viewer()

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    pyramid = data_levels[start_level:] if stop_level == -1 else data_levels[start_level:stop_level]
    if in_memory:
        try:
            pyramid = [p.compute() for p in pyramid]
        except Exception as exc:
            raise RuntimeError(f"Failed to load data into memory: {exc}") from exc

    data_type = str(metadata.get("data_type", "intensity")).lower()
    scale = _axis_scale(metadata, axes, start_level, drop_channel=("c" in axes and data_type != "label"))

    if data_type == "label":
        viewer.add_labels(
            pyramid,
            name=metadata.get("name", "labels"),
            scale=scale,
            metadata=metadata,
            multiscale=True,
        )
        _set_axis_labels(viewer, axes, drop_channel=False)
        return viewer

    add_kwargs = {
        "scale": scale,
        "metadata": metadata,
        "multiscale": True,
    }

    try:
        max_val = da.max(data_levels[-1]).compute()
        min_val = da.min(data_levels[-1]).compute()
        add_kwargs["contrast_limits"] = [max(0, min_val), max(1, int(2 * max_val))]
    except Exception:
        pass

    if "c" in axes:
        c_axis = axes.index("c")
        num_channels = int(data_levels[0].shape[c_axis])
        channel_names = metadata.get("channel_names") or [f"ch_{i}" for i in range(num_channels)]
        channel_colors = metadata.get("channel_colors") or []
        if channel_colors:
            try:
                add_kwargs["colormap"] = [_parse_color(c) for c in channel_colors]
            except Exception:
                add_kwargs["colormap"] = ["gray"] * num_channels
        else:
            add_kwargs["colormap"] = ["gray"] * num_channels
        viewer.add_image(
            pyramid,
            name=channel_names,
            channel_axis=c_axis,
            **add_kwargs,
        )
        _set_axis_labels(viewer, axes, drop_channel=True)
    else:
        viewer.add_image(
            pyramid,
            name=metadata.get("name", "image"),
            colormap="gray",
            **add_kwargs,
        )
        _set_axis_labels(viewer, axes, drop_channel=False)

    return viewer
