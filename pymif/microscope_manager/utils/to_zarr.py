from __future__ import annotations

from pathlib import Path
from typing import Sequence

import dask.array as da
import zarr

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    ZarrWriteConfig,
    _build_multiscales,
    _ngff_payload_parts,
    _normalize_metadata_for_write,
    _resolve_format,
    _set_dimension_names,
    _set_group_ngff_metadata,
    _validate_metadata,
    _write_pyramid_v2,
    _write_pyramid_v3,
)


def to_zarr(
    path: str | Path,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
):
    """Write a pyramid of dask arrays to an OME-Zarr root group.

    ``metadata['data_type']`` controls whether the output is an intensity image
    or a label dataset. Missing ``data_type`` defaults to ``"intensity"``.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _normalize_metadata_for_write(
        metadata,
        axes,
        data_type=cfg.data_type,
    )
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

    _finalize_multiscale_group(
        root,
        data_levels=data_levels,
        metadata=effective_metadata,
        axes=axes,
        ngff_version=ngff_version,
        zarr_format=zarr_format,
        name=effective_metadata.get("name") or "dataset",
        image_label_source="../",
    )

    return root if cfg.compute else delayed


def write_multiscale_to_group(
    group: zarr.Group,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
    name: str | None = None,
    image_label_source: str | None = None,
):
    """Write a Dask pyramid into an existing zarr group.

    Used by :class:`pymif.microscope_manager.ZarrManager` for raw data, image
    subgroups and label groups. Label-vs-image behavior is determined from
    ``metadata['data_type']``.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _normalize_metadata_for_write(
        metadata,
        axes,
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

    _finalize_multiscale_group(
        group,
        data_levels=data_levels,
        metadata=effective_metadata,
        axes=axes,
        ngff_version=ngff_version,
        zarr_format=zarr_format,
        name=name,
        image_label_source=image_label_source,
    )

    return group if cfg.compute else delayed


def _finalize_multiscale_group(
    group: zarr.Group,
    *,
    data_levels: Sequence[da.Array],
    metadata: dict,
    axes: tuple[str, ...],
    ngff_version: str,
    zarr_format: int,
    name: str | None,
    image_label_source: str | None,
) -> None:
    """Write the shared NGFF metadata footer for a populated pyramid group."""
    data_type = normalize_data_type(metadata.get("data_type"))
    multiscales = _build_multiscales(
        metadata,
        axes,
        name=name,
        n_levels=len(data_levels),
    )
    _set_dimension_names(group, multiscales["datasets"], axes, zarr_format=zarr_format)
    omero, extra = _ngff_payload_parts(
        data_levels[0],
        metadata,
        axes,
        image_label_source=image_label_source,
    )
    _set_group_ngff_metadata(
        group,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )
