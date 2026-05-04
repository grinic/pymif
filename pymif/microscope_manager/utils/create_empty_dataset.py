from __future__ import annotations

from typing import Any

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
    _write_empty_pyramid_arrays,
)


def create_empty_dataset(
    root: zarr.Group,
    metadata: dict[str, Any],
    ngff_version: str | None = None,
    zarr_format: int | None = None,
    compressor: str | None = None,
    compressor_level: int = 3,
    data_type: str | None = None,
):
    """Create an on-disk empty OME-Zarr image pyramid from metadata only.

    This is used by :class:`pymif.microscope_manager.zarr_manager.ZarrManager`
    when a new zarr store is opened in append/write mode with a metadata
    dictionary but without image payload yet.
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty dataset.")

    cfg = ZarrWriteConfig(
        ngff_version=ngff_version or metadata.get("ngff_version") or "0.5",
        zarr_format=zarr_format or metadata.get("zarr_format"),
        compressor=compressor,
        compressor_level=compressor_level,
        data_type=data_type or metadata.get("data_type"),
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    sizes = [tuple(s) for s in metadata["size"]]
    chunks = [tuple(c) for c in metadata["chunksize"]]
    dtype = metadata.get("dtype", "uint16")
    axes = normalize_axes(metadata.get("axes"), ndim=len(sizes[0]))

    effective_metadata = _normalize_metadata_for_write(
        metadata,
        axes,
        data_type=cfg.data_type,
    )
    dummy_levels = [da.empty(shape=s, dtype=dtype, chunks=c) for s, c in zip(sizes, chunks)]
    _validate_metadata(dummy_levels, effective_metadata, axes)

    _write_empty_pyramid_arrays(
        root,
        sizes=sizes,
        chunks=chunks,
        dtype=dtype,
        zarr_format=zarr_format,
        compressor=compressor,
        compressor_level=compressor_level,
    )

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    multiscales = _build_multiscales(
        effective_metadata,
        axes,
        name=effective_metadata.get("name") or "OME-Zarr image",
        n_levels=len(dummy_levels),
    )
    _set_dimension_names(root, multiscales["datasets"], axes, zarr_format=zarr_format)
    omero, extra = _ngff_payload_parts(
        dummy_levels[0],
        effective_metadata,
        axes,
        image_label_source="../",
    )

    _set_group_ngff_metadata(
        root,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )
