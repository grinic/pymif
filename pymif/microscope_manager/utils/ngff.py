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

# Axes that `shards="auto"` never merges chunks along by default. Timepoints
# and channels are typically read/processed independently, so keeping each
# one in its own shard (or unsharded chunk) preserves that access pattern;
# the spatial axes (z/y/x) get consolidated automatically.
DEFAULT_SHARD_EXCLUDE_AXES: tuple[str, ...] = ("t", "c")


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
    chunks
        Requested on-disk chunk shape, overriding whatever chunking the input
        dask arrays already have. One of:

        - ``None`` (default): write each level with its existing dask
          chunking, unchanged.
        - A single chunk shape matching the array's ndim: used for every
          level, clipped so no chunk axis exceeds that level's extent.
        - A sequence of chunk shapes, one per pyramid level.

        Applies to both zarr v2 and v3. This only changes the write-time
        chunk shape; it does not touch ``metadata['chunksize']``, which is
        only used when a pyramid is built via
        :func:`~pymif.microscope_manager.utils.pyramid.build_pyramid`.
    shards
        Zarr v3 sharding configuration, applied per pyramid level. One of:

        - ``None`` (default): no sharding, one file per chunk (unchanged
          behaviour).
        - ``"auto"``: pick a shard shape per level automatically so each
          shard is close to ``shard_target_mb`` in (uncompressed) size,
          grouping whole chunks along the axes with the most chunks first.
          Levels that are already small (few chunks, or a chunk already at
          or above the target size) are left unsharded.
        - A single shape tuple matching the array's ndim: used as the
          requested shard shape for every level, snapped to the nearest
          multiple of that level's chunk shape and clipped so it never
          exceeds the level's extent.
        - A sequence of shape tuples, one per pyramid level, each resolved
          the same way.

        Only valid when writing zarr v3 (``ngff_version="0.5"``); sharding
        has no zarr v2 equivalent. ``shard_exclude_axes`` only constrains
        ``"auto"``; an explicit shard shape always applies exactly as given.
    shard_target_mb
        Target *uncompressed* shard size in megabytes used by
        ``shards="auto"``.
    shard_exclude_axes
        Axis names that ``shards="auto"`` never merges chunks along, even if
        doing so would help reach ``shard_target_mb``. Defaults to
        ``("t", "c")`` so each timepoint and channel stays independently
        addressable; the spatial axes (``z``/``y``/``x``) are consolidated
        automatically. Pass ``()`` to allow every axis to grow.
    """

    ngff_version: Literal["0.4", "0.5"] | None = None
    zarr_format: Literal[2, 3] | None = None
    overwrite: bool = True
    compute: bool = True
    storage_options: dict[str, Any] | None = None
    compressor: Literal["blosc", "gzip"] | None = None
    compressor_level: int = 3
    data_type: Literal["intensity", "label"] | None = None
    chunks: Sequence[int] | Sequence[Sequence[int]] | None = None
    shards: Literal["auto"] | Sequence[int] | Sequence[Sequence[int]] | None = None
    shard_target_mb: float = 5 * 1024.0 # in MB
    shard_exclude_axes: Sequence[str] = DEFAULT_SHARD_EXCLUDE_AXES

def _infer_ngff_version(group: zarr.Group) -> str:
    """Infer the NGFF metadata layout used by an existing group."""
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("version", "0.5")
    return "0.4"

def _label_entry(label_name: str) -> str:
    return label_name


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

def _register_label_on_labels_group(root: zarr.Group, label_name: str, ngff_version: str) -> None:
    """Register a label image on the ``labels`` container group."""
    labels_group = root.require_group("labels")
    attrs = labels_group.attrs.asdict()
    entry = _label_entry(label_name)

    if ngff_version == "0.5":
        ome = dict(attrs.get("ome", {}))
        ome.setdefault("version", "0.5")
        labels = list(ome.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        ome["labels"] = labels
        labels_group.attrs["ome"] = ome

        root_attrs = root.attrs.asdict()
        root_ome = root_attrs.get("ome")
        if isinstance(root_ome, dict) and "labels" in root_ome:
            root_ome = dict(root_ome)
            root_ome.pop("labels", None)
            root.attrs["ome"] = root_ome
    else:
        labels = list(attrs.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        labels_group.attrs["labels"] = labels

        if "labels" in root.attrs.asdict():
            del root.attrs["labels"]


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


def _rechunk_to_shape(arr: da.Array, chunks: Sequence[int]) -> da.Array:
    """Rechunk ``arr`` to ``chunks``, clipped so no axis exceeds the array's extent."""
    normalized = tuple(
        max(1, min(int(c), int(s))) for c, s in zip(chunks, arr.shape)
    )
    if normalized == tuple(int(c) for c in _get_chunks(arr)):
        return arr
    return arr.rechunk(normalized)


def _resolve_write_chunks(
    data_levels: Sequence[da.Array],
    chunks: Sequence[int] | Sequence[Sequence[int]] | None,
) -> list[da.Array]:
    """Rechunk each pyramid level to the requested write-time ``chunks``.

    ``chunks`` is either ``None`` (levels are written with whatever chunking
    they already have), a single chunk shape applied to every level, or one
    chunk shape per level. Mirrors how ``shards`` accepts a single shape or a
    per-level list.
    """
    if chunks is None:
        return list(data_levels)

    n_levels = len(data_levels)
    ndim = data_levels[0].ndim

    flat = _shape_tuple(chunks, ndim)
    if flat is not None:
        return [_rechunk_to_shape(arr, flat) for arr in data_levels]

    if len(chunks) != n_levels:
        raise ValueError(
            "chunks must be a single chunk shape or contain one entry per "
            f"pyramid level ({n_levels}), got {len(chunks)}."
        )
    return [_rechunk_to_shape(arr, c) for arr, c in zip(data_levels, chunks)]


def _write_pyramid_v2(
    *,
    root: zarr.Group,
    data_levels: Sequence[da.Array],
    cfg: ZarrWriteConfig,
):
    """Create and populate zarr v2 arrays for each pyramid level."""
    if cfg.shards is not None:
        raise ValueError(
            "Sharding is only supported for zarr_format=3 (NGFF v0.5) datasets; "
            "got zarr_format=2."
        )

    data_levels = _resolve_write_chunks(data_levels, cfg.chunks)
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
    axes: Sequence[str] | None = None,
):
    """Create and populate zarr v3 arrays for each pyramid level."""
    data_levels = _resolve_write_chunks(data_levels, cfg.chunks)
    delayed = []

    chunks_per_level = [_get_chunks(arr) for arr in data_levels]
    shard_shapes = _resolve_shards_for_levels(
        [tuple(arr.shape) for arr in data_levels],
        chunks_per_level,
        data_levels[0].dtype,
        cfg.shards,
        zarr_format=3,
        target_bytes=int(cfg.shard_target_mb * 1024 * 1024),
        axes=axes,
        exclude_axes=cfg.shard_exclude_axes,
    )

    for i, arr in enumerate(data_levels):
        chunks = chunks_per_level[i]

        create_kwargs = {
            "name": str(i),
            "shape": arr.shape,
            "dtype": arr.dtype,
            "chunks": chunks,
        }

        if shard_shapes[i] is not None:
            create_kwargs["shards"] = shard_shapes[i]

        compressors = _build_v3_compressors(cfg.compressor, cfg.compressor_level)
        create_kwargs["compressors"] = compressors

        if cfg.storage_options is not None:
            create_kwargs.update(cfg.storage_options)

        z = root.create_array(**create_kwargs)

        # When a level is sharded, multiple dask chunks land in the same
        # on-disk shard file, so concurrent unlocked writes can race (zarr
        # partial-encodes the shard on every chunk write). Serialize writes
        # for sharded levels; unsharded levels keep the fast unlocked path.
        task = da.store(arr, z, lock=(shard_shapes[i] is not None), compute=cfg.compute)
        if not cfg.compute:
            delayed.append(task)

    return delayed


def _shape_tuple(value: Any, ndim: int) -> tuple[int, ...] | None:
    """Return a positive shape tuple of length ``ndim`` from ``value``, or ``None``."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        candidate = tuple(int(v) for v in value)
    except TypeError:
        return None
    if len(candidate) != ndim or any(v <= 0 for v in candidate):
        return None
    return candidate


def _auto_shard_for_level(
    shape: Sequence[int],
    chunk: Sequence[int],
    itemsize: int,
    target_bytes: int,
    *,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] = (),
) -> tuple[int, ...] | None:
    """Pick a shard shape for one pyramid level targeting ``target_bytes`` per shard.

    Whole chunks are grouped into a shard, growing the axes with the most
    available chunks first (typically the spatial ``z``/``y``/``x`` axes).
    Axes whose name is in ``exclude_axes`` (requires ``axes`` to be given)
    are never grown, so e.g. timepoints or channels can be kept one-per-shard
    even while z/y/x is merged. Levels with no growable axis holding more
    than one chunk, or whose chunk is already at or above the target size,
    are left unsharded (``None``).
    """
    ndim = len(chunk)
    chunks_per_axis = [max(1, -(-int(s) // int(c))) for s, c in zip(shape, chunk)]

    if axes is not None and exclude_axes:
        growable = [i for i in range(ndim) if axes[i] not in exclude_axes]
    else:
        growable = list(range(ndim))

    if not growable or all(chunks_per_axis[i] <= 1 for i in growable):
        return None

    chunk_bytes = itemsize
    for c in chunk:
        chunk_bytes *= int(c)
    if chunk_bytes <= 0 or chunk_bytes >= target_bytes:
        return None

    want_chunks = max(2, round(target_bytes / chunk_bytes))

    multiplier = [1] * ndim
    order = sorted(growable, key=lambda a: chunks_per_axis[a], reverse=True)

    def _total() -> int:
        total = 1
        for m in multiplier:
            total *= m
        return total

    progressed = True
    while _total() < want_chunks and progressed:
        progressed = False
        for axis in order:
            if multiplier[axis] < chunks_per_axis[axis]:
                multiplier[axis] += 1
                progressed = True
                if _total() >= want_chunks:
                    break

    if all(m == 1 for m in multiplier):
        return None

    return tuple(int(c) * int(m) for c, m in zip(chunk, multiplier))


def _snap_shard_shape(
    spec: Any,
    shape: Sequence[int],
    chunk: Sequence[int],
) -> tuple[int, ...] | None:
    """Round a requested shard shape to a valid multiple of ``chunk``.

    The request is rounded to the nearest whole number of chunks per axis
    (minimum one) and clipped so the shard never spans more chunks than the
    level actually has. Returns ``None`` when the result collapses back to
    the plain chunk shape (i.e. sharding would add no value at this level).
    """
    ndim = len(chunk)
    parsed = _shape_tuple(spec, ndim)
    if parsed is None:
        raise ValueError(
            f"Invalid shard shape {spec!r}; expected a length-{ndim} sequence "
            "of positive ints."
        )

    out = []
    for value, c, s in zip(parsed, chunk, shape):
        c = int(c)
        s = int(s)
        chunks_per_axis = max(1, -(-s // c))
        mult = min(max(1, round(value / c)), chunks_per_axis)
        out.append(c * mult)

    result = tuple(out)
    if result == tuple(int(c) for c in chunk):
        return None
    return result


def _resolve_shards_for_levels(
    shapes: Sequence[Sequence[int]],
    chunks: Sequence[Sequence[int]],
    dtype: Any,
    shards: Literal["auto"] | Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    zarr_format: int,
    target_bytes: int,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] = (),
) -> list[tuple[int, ...] | None]:
    """Resolve a per-pyramid-level shard shape (or ``None``) for zarr v3 writes.

    ``axes``/``exclude_axes`` only affect ``shards="auto"``; an explicit
    shard shape (single tuple or per-level list) always applies exactly as
    requested.
    """
    n_levels = len(shapes)
    if shards is None:
        return [None] * n_levels

    if zarr_format != 3:
        raise ValueError(
            "Sharding is only supported for zarr_format=3 (NGFF v0.5) datasets; "
            f"got zarr_format={zarr_format}."
        )

    ndim = len(chunks[0])
    itemsize = np.dtype(dtype).itemsize

    if isinstance(shards, str):
        if shards != "auto":
            raise ValueError(
                f"Unsupported shards value {shards!r}; use 'auto', an explicit "
                "shard shape, or one shard shape per pyramid level."
            )
        return [
            _auto_shard_for_level(
                shape, chunk, itemsize, target_bytes,
                axes=axes, exclude_axes=exclude_axes,
            )
            for shape, chunk in zip(shapes, chunks)
        ]

    flat = _shape_tuple(shards, ndim)
    if flat is not None:
        return [_snap_shard_shape(flat, shape, chunk) for shape, chunk in zip(shapes, chunks)]

    if len(shards) != n_levels:
        raise ValueError(
            "shards must be 'auto', a single shard shape, or contain one "
            f"entry per pyramid level ({n_levels}), got {len(shards)}."
        )
    return [
        _snap_shard_shape(spec, shape, chunk)
        for spec, shape, chunk in zip(shards, shapes, chunks)
    ]


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
