from __future__ import annotations

from typing import Any

import dask.array as da
import zarr

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    ZarrWriteConfig,
    _build_axes,
    _build_coordinate_transformations,
    _build_omero_metadata,
    _build_v2_compressor,
    _build_v3_compressors,
    _resolve_format,
    _set_group_ngff_metadata,
    _set_dimension_names,
    _validate_metadata,
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

    effective_metadata = dict(metadata)
    effective_metadata["axes"] = "".join(axes)
    effective_metadata["data_type"] = normalize_data_type(cfg.data_type or metadata.get("data_type"))

    dummy_levels = [da.empty(shape=s, dtype=dtype, chunks=c) for s, c in zip(sizes, chunks)]
    _validate_metadata(dummy_levels, effective_metadata, axes)

    for i, (shape, chunk) in enumerate(zip(sizes, chunks)):
        kwargs = {
            "name": str(i),
            "shape": shape,
            "chunks": chunk,
            "dtype": dtype,
        }
        if zarr_format == 2:
            kwargs["compressor"] = _build_v2_compressor(compressor, compressor_level)
            kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
        else:
            compressors = _build_v3_compressors(compressor, compressor_level)
            if compressors is not None:
                kwargs["compressors"] = compressors

        root.create_array(**kwargs)

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    multiscales = {
        "name": metadata.get("name", "OME-Zarr image"),
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

    _set_dimension_names(root, multiscales["datasets"], axes, zarr_format=zarr_format)

    omero = None
    extra = None
    if data_type == "intensity":
        omero = _build_omero_metadata(dummy_levels[0], axes, effective_metadata)
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