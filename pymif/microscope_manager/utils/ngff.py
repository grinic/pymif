import zarr
import numpy as np
import dask.array as da
from typing import Any, Literal, Sequence
from dataclasses import dataclass
from numcodecs import Blosc, GZip

DEFAULT_COLORS = (
    "FF0000", "00FF00", "0000FF", "FFFF00",
    "FF00FF", "00FFFF", "FFFFFF", "808080",
)
SPATIAL_AXES = {"z", "y", "x"}

@dataclass(slots=True)
class ZarrWriteConfig:
    ngff_version: Literal["0.4", "0.5"] | None = None
    zarr_format: Literal[2, 3] | None = None
    overwrite: bool = True
    compute: bool = True
    storage_options: dict[str, Any] | None = None
    compressor: Literal["blosc", "gzip"] | None = None
    compressor_level: int = 3

def _infer_ngff_version(group: zarr.Group) -> str:
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("version", "0.5")
    return "0.4"

def _register_label_on_root(root: zarr.Group, label_name: str, ngff_version: str) -> None:
    label_path = f"labels/{label_name}"
    attrs = root.attrs.asdict()

    if ngff_version == "0.5":
        ome = dict(attrs.get("ome", {}))
        labels = list(ome.get("labels", []))
        if label_path not in labels:
            labels.append(label_path)
        ome["labels"] = labels
        root.attrs["ome"] = ome
    else:
        labels = list(attrs.get("labels", []))
        if label_path not in labels:
            labels.append(label_path)
        root.attrs["labels"] = labels

def _get_multiscales(group: zarr.Group) -> list[dict[str, Any]]:
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("multiscales", [])
    return attrs.get("multiscales", [])

def _set_group_ngff_metadata(
    group: zarr.Group,
    *,
    ngff_version: str,
    multiscales: dict[str, Any],
    omero: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    extra = extra or {}

    if ngff_version == "0.5":
        payload = {"version": "0.5", "multiscales": [multiscales]}
        if omero is not None:
            payload["omero"] = omero
        payload.update(extra)
        group.attrs["ome"] = payload
    else:
        group.attrs["multiscales"] = [multiscales]
        if omero is not None:
            group.attrs["omero"] = omero
        for k, v in extra.items():
            group.attrs[k] = v

def _get_group_multiscales(group: zarr.Group):
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("multiscales")
    return attrs.get("multiscales")




def _resolve_format(cfg: ZarrWriteConfig) -> tuple[str, int]:
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
    if hasattr(arr, "chunksize") and arr.chunksize is not None:
        return tuple(int(x) for x in arr.chunksize)
    return tuple(int(c[0]) for c in arr.chunks)


def _build_v2_compressor(compressor: str | None, level: int):
    if compressor is None:
        return None
    if compressor == "blosc":
        return Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)
    if compressor == "gzip":
        return GZip(level=level)
    raise ValueError(f"Unsupported compressor for zarr v2: {compressor}")


def _build_v3_compressors(compressor: str | None, level: int):
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
    ndim = data_levels[0].ndim
    if len(axes) != ndim:
        raise ValueError(f"`axes` has length {len(axes)} but arrays have ndim={ndim}.")

    for arr in data_levels[1:]:
        if arr.ndim != ndim:
            raise ValueError("All pyramid levels must have the same ndim.")

    scales = metadata.get("scales")
    if scales is None or len(scales) != len(data_levels):
        raise ValueError("`metadata['scales']` must contain one entry per pyramid level.")

    spatial_axes = [ax for ax in axes if ax in SPATIAL_AXES]
    for scale in scales:
        if len(scale) != len(spatial_axes):
            raise ValueError(
                "Each scale entry must match the number of spatial axes "
                f"({len(spatial_axes)}), got {len(scale)}."
            )

    spatial_units = list(metadata.get("units", ()))
    if spatial_units and len(spatial_units) != len(spatial_axes):
        raise ValueError(
            "`metadata['units']` must match the number of spatial axes "
            f"({len(spatial_axes)})."
        )

    if "t" in axes and metadata.get("time_increment") is None:
        raise ValueError("A `t` axis requires `metadata['time_increment']`.")


def _build_axes(axes: tuple[str, ...], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    axis_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    spatial_axes = [ax for ax in axes if ax in SPATIAL_AXES]
    spatial_units = [
        _normalize_unit(u) for u in metadata.get("units", [None] * len(spatial_axes))
    ]
    spatial_unit_map = dict(zip(spatial_axes, spatial_units))
    time_unit = _normalize_unit(metadata.get("time_increment_unit"))

    out = []
    for ax in axes:
        entry = {"name": ax, "type": axis_types.get(ax, "unknown")}
        if ax == "t" and time_unit:
            entry["unit"] = time_unit
        elif ax in spatial_unit_map and spatial_unit_map[ax]:
            entry["unit"] = spatial_unit_map[ax]
        out.append(entry)
    return out


def _build_coordinate_transformations(
    *,
    axes: tuple[str, ...],
    scales: Sequence[Sequence[float]],
    time_increment: float | None,
) -> list[list[dict[str, Any]]]:
    out = []
    for spatial_scale in scales:
        spatial_iter = iter(spatial_scale)
        full_scale = []
        for ax in axes:
            if ax == "t":
                full_scale.append(float(time_increment if time_increment is not None else 1.0))
            elif ax == "c":
                full_scale.append(1.0)
            elif ax in SPATIAL_AXES:
                full_scale.append(float(next(spatial_iter)))
            else:
                full_scale.append(1.0)

        out.append([{"type": "scale", "scale": full_scale}])
    return out


def _build_omero_metadata(
    arr: da.Array,
    axes: tuple[str, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
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
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _normalize_color(color: Any) -> str:
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
    if not unit:
        return None

    aliases = {
        "um": "micrometer",
        "μm": "micrometer",
        "\u00b5m": "micrometer",
        "micron": "micrometer",
        "microns": "micrometer",
        "s": "second",
        "sec": "second",
    }
    return aliases.get(unit.strip(), unit.strip())

def _get_group_ome_attrs(group: zarr.Group) -> dict[str, Any]:
    attrs = group.attrs.asdict()
    return attrs.get("ome", attrs)


