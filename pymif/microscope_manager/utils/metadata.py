from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Real
from typing import Any
import re
import warnings

from .axes import normalize_data_type, spatial_axes_in_order

VALID_METADATA_KEYS = {
    "channel_names",
    "channel_colors",
    "scales",
    "time_increment",
    "time_increment_unit",
    "units",
    "data_type",
}

_HEX_COLOR = re.compile(r"^#?[0-9a-fA-F]{6}$")


def parse_channel_color(value: str) -> str:
    """Normalize a matplotlib color name or six-digit hex value to NGFF hex."""
    if not isinstance(value, str):
        raise TypeError("Channel colors must be strings.")

    if _HEX_COLOR.match(value):
        return value.replace("#", "").upper()

    from matplotlib.colors import cnames

    lower = value.lower()
    if lower in cnames:
        return cnames[lower].replace("#", "").upper()

    raise TypeError(
        f"Invalid color {value!r}. Use a 6-digit hex code or a valid matplotlib color name."
    )


def _chunksize(level: Any) -> tuple[int, ...]:
    chunksize = getattr(level, "chunksize", None)
    if chunksize is not None:
        return tuple(chunksize)
    chunks = getattr(level, "chunks", None)
    if chunks is not None:
        return tuple(int(axis_chunks[0]) for axis_chunks in chunks)
    return tuple(int(s) for s in level.shape)


def reorder_channel_axis(
    data_levels: Sequence[Any],
    metadata: Mapping[str, Any],
    new_order: Sequence[int],
    *,
    dataset_name: str = "dataset",
) -> tuple[list[Any], dict[str, Any]]:
    """Reorder the channel axis and keep channel metadata aligned."""
    if not data_levels:
        raise ValueError("No data loaded.")

    out_metadata = dict(metadata)
    axes = str(out_metadata.get("axes", "")).lower()
    if "c" not in axes:
        raise ValueError(f"Dataset {dataset_name!r} has no channel axis to reorder.")

    c_axis = axes.index("c")
    n_channels = int(data_levels[0].shape[c_axis])
    order = [int(i) for i in new_order]
    if sorted(order) != list(range(n_channels)):
        raise ValueError(
            f"new_order must be a permutation of 0..{n_channels - 1} "
            f"for dataset {dataset_name!r}."
        )

    reordered = []
    for level in data_levels:
        slicer = [slice(None)] * level.ndim
        slicer[c_axis] = order
        reordered.append(level[tuple(slicer)])

    for key in ("channel_names", "channel_colors"):
        values = out_metadata.get(key)
        if values:
            out_metadata[key] = [values[i] for i in order]

    out_metadata["size"] = [tuple(level.shape) for level in reordered]
    out_metadata["chunksize"] = [_chunksize(level) for level in reordered]
    return reordered, out_metadata


def apply_metadata_updates(
    data_levels: Sequence[Any],
    metadata: MutableMapping[str, Any],
    updates: Mapping[str, Any],
    *,
    dataset_name: str = "dataset",
) -> list[str]:
    """Validate and apply PyMIF metadata updates in-place.

    The same rules are used by ``MicroscopeManager`` and ``ZarrManager`` so
    metadata behavior does not drift between in-memory and on-disk managers.
    """
    updated: list[str] = []
    axes = str(metadata.get("axes", "")).lower()
    spatial_axes = spatial_axes_in_order(axes) if axes else ()

    for key, value in updates.items():
        if key not in VALID_METADATA_KEYS:
            warnings.warn(f"Unsupported or unknown metadata key {key!r}.", stacklevel=2)
            continue

        if key in {"channel_names", "channel_colors"}:
            if "c" not in axes:
                warnings.warn(
                    f"Skipping {key!r} for dataset {dataset_name!r}: no channel axis.",
                    stacklevel=2,
                )
                continue
            expected_channels = int(data_levels[0].shape[axes.index("c")])
            if len(value) != expected_channels:
                warnings.warn(
                    f"Skipping {key!r} for dataset {dataset_name!r}: expected "
                    f"{expected_channels} values, got {len(value)}.",
                    stacklevel=2,
                )
                continue
            if key == "channel_colors":
                value = [parse_channel_color(v) for v in value]

        elif key == "scales":
            if not isinstance(value, list):
                raise TypeError("'scales' must be a list.")
            if len(value) != len(data_levels):
                raise ValueError(
                    f"'scales' must contain one entry per pyramid level for "
                    f"dataset {dataset_name!r}. Expected {len(data_levels)}, got {len(value)}."
                )
            for scale in value:
                if not isinstance(scale, (tuple, list)):
                    raise TypeError("Each scale entry must be a tuple or list.")
                if len(scale) != len(spatial_axes):
                    raise ValueError("Each scale entry must match the dataset spatial axes.")
            value = [tuple(float(v) for v in scale) for scale in value]

        elif key == "time_increment":
            if value is not None and (not isinstance(value, Real) or isinstance(value, bool) or value <= 0):
                raise ValueError("'time_increment' must be a positive number or None.")

        elif key == "time_increment_unit":
            if value is not None and not isinstance(value, str):
                raise TypeError("'time_increment_unit' must be a string or None.")

        elif key == "units":
            if not isinstance(value, (tuple, list)):
                raise TypeError("'units' must be a tuple or list.")
            if len(value) != len(spatial_axes):
                raise ValueError("'units' must match the dataset spatial axes.")
            value = tuple(value)

        elif key == "data_type":
            value = normalize_data_type(value)

        metadata[key] = value
        updated.append(key)

    return updated
