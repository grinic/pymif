from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import dask.array as da
import numpy as np
import zarr
from numcodecs import Blosc, GZip

from .axes import (
    DATA_TYPES,
    SPATIAL_AXIS_SET,
    normalize_axes,
    normalize_data_type,
    spatial_axes_in_order,
)

DEFAULT_COLORS = (
    "FF0000", "00FF00", "0000FF", "FFFF00",
    "FF00FF", "00FFFF", "FFFFFF", "808080",
)
SPATIAL_AXES = SPATIAL_AXIS_SET


@dataclass(slots=True)
class ZarrWriteConfig:
    """Configuration container for NGFF/OME-Zarr writing operations.

    Parameters
    ----------
    ngff_version, zarr_format
        ``0.4``/Zarr v2 and ``0.5``/Zarr v3 are the supported pairs.
    data_type
        Optional dataset semantic type.  Use ``"intensity"`` for regular image
        intensities or ``"label"`` for integer segmentation/annotation data.
    """

    ngff_version: Literal["0.4", "0.5"] | None = None
    zarr_format: Literal[2, 3] | None = None
    overwrite: bool = True
    compute: bool = True
    storage_options: dict[str, Any] | None = None
    compressor: Literal["blosc", "gzip"] | None = None
    compressor_level: int = 3
    data_type: Literal["intensity", "label"] | None = None

def _infer_ngff_version(group: zarr.Group) -> str:
    """Infer the NGFF metadata layout used by an existing group."""
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("version", "0.5")
    return "0.4"

def _label_entry(label_name: str) -> dict[str, Any]:
    label_path = f"labels/{label_name}"
    return label_path


def _labels_contains(labels: Sequence[Any], label_name: str) -> bool:
    label_path = f"labels/{label_name}"
    for item in labels:
        if item == label_name or item == label_path:
            return True
        if isinstance(item, dict) and item.get("name") == label_name:
            return True
        if isinstance(item, dict) and item.get("path") == label_path:
            return True
    return False

def _register_label_on_root(root: zarr.Group, label_name: str, ngff_version: str) -> None:
    """Register a label group in the root label list for the active NGFF version."""
    attrs = root.attrs.asdict()
    entry = _label_entry(label_name)

    if ngff_version == "0.5":
        ome = dict(attrs.get("ome", {}))
        ome.setdefault("version", "0.5")
        labels = list(ome.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        ome["labels"] = labels
        root.attrs["ome"] = ome
    else:
        labels = list(attrs.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        root.attrs["labels"] = labels


def _get_group_ome_attrs(group: zarr.Group) -> dict[str, Any]:
    """Return the effective OME-NGFF metadata mapping for a group."""
    attrs = group.attrs.asdict()
    ome = attrs.get("ome")
    return ome if isinstance(ome, dict) else attrs

def _get_multiscales(group: zarr.Group) -> list[dict[str, Any]]:
    """Return the raw ``multiscales`` list from a group across NGFF versions."""
    return _get_group_ome_attrs(group).get("multiscales", [])

def _get_group_multiscales(group: zarr.Group):
    """Compatibility helper returning the stored multiscales block for a group."""
    return _get_group_ome_attrs(group).get("multiscales")

def _infer_data_type_from_group(group: zarr.Group) -> str:
    """Infer ``intensity`` or ``label`` from explicit and legacy metadata."""
    attrs = group.attrs.asdict()
    image_meta = _get_group_ome_attrs(group)

    explicit = image_meta.get("data_type") or attrs.get("data_type")
    if explicit is not None:
        return normalize_data_type(explicit)

    if "image-label" in image_meta or "image-label" in attrs:
        return "label"

    multiscales = image_meta.get("multiscales", [])
    if multiscales:
        ms_type = multiscales[0].get("type")
        if ms_type in DATA_TYPES or ms_type in {"image", "labels"}:
            return normalize_data_type(ms_type)

    return "intensity"

def _set_group_ngff_metadata(
    group: zarr.Group,
    *,
    ngff_version: str,
    multiscales: dict[str, Any],
    omero: dict[str, Any] | None = None,
    data_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write NGFF metadata to ``group`` using either the v0.4 or v0.5 layout."""
    extra = dict(extra or {})
    normalized_data_type = normalize_data_type(data_type)
    multiscales = dict(multiscales)
    multiscales.setdefault("type", "label" if normalized_data_type == "label" else "image")

    if normalized_data_type == "label":
        extra.setdefault("image-label", {"source": {"image": "../../"}})

    if ngff_version == "0.5":
        payload = {
            "version": "0.5",
            "data_type": normalized_data_type,
            "multiscales": [multiscales],
        }
        if omero is not None and normalized_data_type == "intensity":
            payload["omero"] = omero
        payload.update(extra)
        group.attrs["ome"] = payload
    else:
        group.attrs["data_type"] = normalized_data_type
        group.attrs["multiscales"] = [multiscales]
        if omero is not None and normalized_data_type == "intensity":
            group.attrs["omero"] = omero
        for key, value in extra.items():
            group.attrs[key] = value

def _set_dimension_names(
    group: zarr.Group,
    datasets: Sequence[dict[str, Any]],
    axes: Sequence[str],
    *,
    zarr_format: int,
) -> None:
    """Write Zarr v3 array-level dimension names required by NGFF v0.5."""
    if zarr_format != 3:
        return
    names = [str(axis) for axis in axes]
    for dataset in datasets:
        path = dataset.get("path")
        if path in group:
            group[path].attrs["dimension_names"] = names


def _resolve_format(cfg: ZarrWriteConfig) -> tuple[str, int]:
    """Resolve and validate the NGFF version / zarr format pair to use."""
    ngff_version = cfg.ngff_version or ("0.5" if cfg.zarr_format in (None, 3) else "0.4")
    zarr_format = cfg.zarr_format or (3 if ngff_version == "0.5" else 2)

    if (ngff_version, zarr_format) not in {("0.4", 2), ("0.5", 3)}:
        raise ValueError(
            f"Incompatible ngff_version/zarr_format pair: {ngff_version}/{zarr_format}. "
            "Use 0.4 with zarr v2 or 0.5 with zarr v3."
        )

    return ngff_version, zarr_format


def _write_pyramid_v2(
    *,
    root: zarr.Group,
    data_levels: Sequence[da.Array],
    cfg: ZarrWriteConfig,
):
    """Create and populate zarr v2 arrays for each pyramid level."""
    delayed = []

    for i, arr in enumerate(data_levels):
        chunks = _get_chunks(arr)

        create_kwargs = {
            "name": str(i),
            "shape": arr.shape,
            "dtype": arr.dtype,
            "chunks": chunks,
            "compressor": _build_v2_compressor(cfg.compressor, cfg.compressor_level),
            "chunk_key_encoding": {"name": "v2", "separator": "/"},
        }

        if cfg.storage_options is not None:
            create_kwargs.update(cfg.storage_options)

        z = root.create_array(**create_kwargs)

        task = da.store(arr, z, lock=False, compute=cfg.compute)
        if not cfg.compute:
            delayed.append(task)

    return delayed


def _write_pyramid_v3(
    *,
    root: zarr.Group,
    data_levels: Sequence[da.Array],
    cfg: ZarrWriteConfig,
):
    """Create and populate zarr v3 arrays for each pyramid level."""
    delayed = []

    for i, arr in enumerate(data_levels):
        chunks = _get_chunks(arr)

        create_kwargs = {
            "name": str(i),
            "shape": arr.shape,
            "dtype": arr.dtype,
            "chunks": chunks,
        }

        compressors = _build_v3_compressors(cfg.compressor, cfg.compressor_level)
        if compressors is not None:
            create_kwargs["compressors"] = compressors

        if cfg.storage_options is not None:
            create_kwargs.update(cfg.storage_options)

        z = root.create_array(**create_kwargs)

        task = da.store(arr, z, lock=False, compute=cfg.compute)
        if not cfg.compute:
            delayed.append(task)

    return delayed


def _get_chunks(arr: da.Array) -> tuple[int, ...]:
    """Return one normalized chunk tuple for a dask array."""
    if hasattr(arr, "chunksize") and arr.chunksize is not None:
        return tuple(int(x) for x in arr.chunksize)
    return tuple(int(c[0]) for c in arr.chunks)


def _build_v2_compressor(compressor: str | None, level: int):
    """Construct a zarr v2-compatible compressor configuration."""
    if compressor is None:
        return None
    if compressor == "blosc":
        return Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)
    if compressor == "gzip":
        return GZip(level=level)
    raise ValueError(f"Unsupported compressor for zarr v2: {compressor}")


def _build_v3_compressors(compressor: str | None, level: int):
    """Construct a zarr v3-compatible compressor chain."""
    if compressor is None:
        return None
    if compressor == "blosc":
        return [
            zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=level,
                shuffle=zarr.codecs.BloscShuffle.bitshuffle,
            )
        ]
    raise ValueError(f"Unsupported compressor for zarr v3: {compressor}")


def _validate_metadata(
    data_levels: Sequence[da.Array],
    metadata: dict[str, Any],
    axes: tuple[str, ...],
) -> None:
    """Validate the minimal metadata contract required for NGFF writing."""
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    ndim = data_levels[0].ndim
    axes = normalize_axes(axes, ndim=ndim)

    for arr in data_levels[1:]:
        if arr.ndim != ndim:
            raise ValueError("All pyramid levels must have the same ndim.")

    data_type = normalize_data_type(metadata.get("data_type"))
    if data_type == "label" and not np.issubdtype(np.dtype(data_levels[0].dtype), np.integer):
        raise ValueError("Label datasets must use an integer dtype.")

    sizes = metadata.get("size")
    if sizes is not None and len(sizes) != len(data_levels):
        raise ValueError("metadata['size'] must contain one entry per pyramid level.")

    chunks = metadata.get("chunksize")
    if chunks is not None and len(chunks) != len(data_levels):
        raise ValueError("metadata['chunksize'] must contain one entry per pyramid level.")

    scales = metadata.get("scales")
    if scales is None or len(scales) != len(data_levels):
        raise ValueError("metadata['scales'] must contain one entry per pyramid level.")

    spatial_axes = spatial_axes_in_order(axes)
    for scale in scales:
        if len(scale) != len(spatial_axes):
            raise ValueError(
                "Each scale entry must match the number of spatial axes "
                f"({len(spatial_axes)}), got {len(scale)}."
            )

    spatial_units = list(metadata.get("units", ()))
    if spatial_units and len(spatial_units) != len(spatial_axes):
        raise ValueError(
            "metadata['units'] must match the number of spatial axes "
            f"({len(spatial_axes)})."
        )

    if "t" in axes and metadata.get("time_increment") is None:
        raise ValueError("A 't' axis requires metadata['time_increment'].")


def _build_axes(axes: tuple[str, ...], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the NGFF ``axes`` description from the normalized PyMIF metadata."""
    axes = normalize_axes(axes)
    axis_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    spatial_labels = spatial_axes_in_order(axes)
    spatial_units = [
        _normalize_unit(u) for u in metadata.get("units", [None] * len(spatial_labels))
    ]
    spatial_unit_map = dict(zip(spatial_labels, spatial_units))
    time_unit = _normalize_unit(metadata.get("time_increment_unit"))

    out = []
    for ax in axes:
        entry = {"name": ax, "type": axis_types[ax]}
        if ax == "t" and time_unit:
            entry["unit"] = time_unit
        elif ax in spatial_unit_map and spatial_unit_map[ax]:
            entry["unit"] = spatial_unit_map[ax]
        out.append(entry)
    return out


def _build_coordinate_transformations(
    *,
    axes: tuple[str, ...] | str,
    scales: Sequence[Sequence[float]],
    time_increment: float | None,
) -> list[list[dict[str, Any]]]:
    """Generate one NGFF scale transformation entry per pyramid level."""
    axes = normalize_axes(axes)
    out = []
    for spatial_scale in scales:
        spatial_iter = iter(spatial_scale)
        full_scale = []
        for ax in axes:
            if ax == "t":
                full_scale.append(float(time_increment if time_increment is not None else 1.0))
            elif ax == "c":
                full_scale.append(1.0)
            elif ax in SPATIAL_AXIS_SET:
                full_scale.append(float(next(spatial_iter)))
            else:
                full_scale.append(1.0)
        out.append([{"type": "scale", "scale": full_scale}])
    return out


def _build_omero_metadata(
    arr: da.Array,
    axes: tuple[str, ...] | str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create OMERO channel display metadata for an intensity array."""
    axes = normalize_axes(axes, ndim=arr.ndim)
    c_size = arr.shape[axes.index("c")] if "c" in axes else 1
    ch_names = list(metadata.get("channel_names") or [])
    ch_colors = list(metadata.get("channel_colors") or [])

    lo, hi = _default_window(arr.dtype)
    channels = []
    for i in range(c_size):
        label = ch_names[i] if i < len(ch_names) else f"channel_{i}"
        color = ch_colors[i] if i < len(ch_colors) else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        channels.append(
            {
                "label": label,
                "color": _normalize_color(color),
                "window": {"start": lo, "end": hi, "min": lo, "max": hi},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
        )

    return {"channels": channels, "rdefs": {"model": "color"}}


def _default_window(dtype: np.dtype | str) -> tuple[float, float]:
    """Return a default display range for the provided dtype."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.bool_):
        return 0.0, 1.0
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _normalize_color(color: Any) -> str:
    """Normalize different color inputs to a six-digit uppercase hex string."""
    if isinstance(color, int):
        return f"{color & 0xFFFFFF:06X}"

    if isinstance(color, str):
        value = color.strip().lstrip("#")
        if value.lower().startswith("0x"):
            value = value[2:]
        if len(value) == 6:
            return value.upper()

    return "FFFFFF"


def _normalize_unit(unit: str | None) -> str | None:
    """Map common unit aliases to names expected in NGFF metadata."""
    if not unit:
        return None

    aliases = {
        "um": "micrometer",
        "micron": "micrometer",
        "microns": "micrometer",
        "s": "second",
        "sec": "second",
    }
    unit = str(unit).strip()
    return aliases.get(unit, unit)
