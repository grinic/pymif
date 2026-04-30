from __future__ import annotations

from pathlib import Path
from typing import Sequence

import dask.array as da
import zarr

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    ZarrWriteConfig,
    _build_axes,
    _build_coordinate_transformations,
    _build_omero_metadata,
    _resolve_format,
    _set_group_ngff_metadata,
    _set_dimension_names,
    _validate_metadata,
    _write_pyramid_v2,
    _write_pyramid_v3,
)

def _metadata_for_write(
    metadata: dict,
    axes: tuple[str, ...],
    *,
    config: ZarrWriteConfig,
    is_label: bool | None = None,
) -> dict:
    out = dict(metadata)
    data_type = normalize_data_type(config.data_type or metadata.get("data_type"), is_label=is_label)
    out["axes"] = "".join(axes)
    out["data_type"] = data_type
    return out

def _build_multiscales(
    metadata: dict,
    axes: tuple[str, ...],
    *,
    name: str | None,
    n_levels: int,
) -> dict:
    data_type = normalize_data_type(metadata.get("data_type"))
    return {
        "name": name or metadata.get("name") or "dataset",
        "axes": _build_axes(axes, metadata),
        "datasets": [
            {
                "path": str(i),
                "coordinateTransformations": ct,
            }
            for i, ct in enumerate(
                _build_coordinate_transformations(
                    axes=axes,
                    scales=metadata["scales"],
                    time_increment=metadata.get("time_increment"),
                )
            )
        ],
        "type": "label" if data_type == "label" else "image",
    }

def to_zarr(
    path: str | Path,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
):
    """Write a pyramid of dask arrays to an OME-Zarr root group.

    Parameters
    ----------
    path : str | Path
        Destination zarr store.
    data_levels : sequence of dask.array.Array
        Pyramid levels ordered from finest to coarsest resolution.
    metadata : dict
        Normalized PyMIF metadata dictionary describing axes, scales, channel
        metadata and units.
    config : ZarrWriteConfig | None
        Output configuration controlling NGFF version, zarr format, overwrite
        behaviour and compression.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _metadata_for_write(metadata, axes, config=cfg)
    _validate_metadata(data_levels, effective_metadata, axes)

    ngff_version, zarr_format = _resolve_format(cfg)
    root = zarr.open_group(
        str(Path(path)),
        mode="w" if cfg.overwrite else "w-",
        zarr_format=zarr_format,
    )

    if zarr_format == 3:
        delayed = _write_pyramid_v3(root=root, data_levels=data_levels, cfg=cfg)
    else:
        delayed = _write_pyramid_v2(root=root, data_levels=data_levels, cfg=cfg)

    multiscales = _build_multiscales(
        effective_metadata,
        axes,
        name=effective_metadata.get("name") or "dataset",
        n_levels=len(data_levels),
    )

    _set_dimension_names(root, multiscales["datasets"], axes, zarr_format=zarr_format)

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    omero = None
    extra = None
    if data_type == "intensity":
        omero = _build_omero_metadata(data_levels[0], axes, effective_metadata)
    else:
        extra = {"image-label": {"source": {"image": "../"}}}

    _set_group_ngff_metadata(
        root,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )

    return root if cfg.compute else delayed


def write_multiscale_to_group(
    group: zarr.Group,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
    name: str | None = None,
    is_label: bool = False,
):
    """Write a pyramid of Dask arrays into an existing zarr group.

    Used by :class:`pymif.microscope_manager.ZarrManager` for raw data, image
    subgroups and label groups.  The axes may be any subset of ``tczyx``.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _metadata_for_write(
        metadata,
        axes,
        config=cfg,
        is_label=True if is_label else None,
    )
    _validate_metadata(data_levels, effective_metadata, axes)

    ngff_version, zarr_format = _resolve_format(cfg)

    if cfg.overwrite:
        for key in list(group.array_keys()):
            if str(key).isdigit():
                del group[key]

    if zarr_format == 3:
        delayed = _write_pyramid_v3(root=group, data_levels=data_levels, cfg=cfg)
    else:
        delayed = _write_pyramid_v2(root=group, data_levels=data_levels, cfg=cfg)

    multiscales = _build_multiscales(
        effective_metadata,
        axes,
        name=name,
        n_levels=len(data_levels),
    )

    _set_dimension_names(group, multiscales["datasets"], axes, zarr_format=zarr_format)

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    omero = None
    extra = None
    if data_type == "intensity":
        omero = _build_omero_metadata(data_levels[0], axes, effective_metadata)
    else:
        label_source = "../../" if is_label else "../"
        extra = {"image-label": {"source": {"image": label_source}}}

    _set_group_ngff_metadata(
        group,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )

    return group if cfg.compute else delayed
