from __future__ import annotations

from typing import Any

import dask.array as da
import zarr

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    ZarrWriteConfig,
    _build_multiscales,
    _ngff_payload_parts,
    _infer_ngff_version,
    _normalize_metadata_for_write,
    _register_label_on_root,
    _resolve_format,
    _set_dimension_names,
    _set_group_ngff_metadata,
    _validate_metadata,
    _write_empty_pyramid_arrays,
)


def create_empty_group(
    root: zarr.Group,
    group_name: str,
    metadata: dict[str, Any],
    ngff_version: str | None = None,
    zarr_format: int | None = None,
    compressor: str | None = None,
    compressor_level: int = 3,
    data_type: str | None = None,
):
    """Create an empty image subgroup or label subgroup inside an existing root.

    Label groups are requested by setting ``metadata['data_type'] = 'label'``
    (or by passing the optional ``data_type='label'`` override). The function no
    longer rewrites intensity image metadata when creating labels; callers must
    provide explicit label axes, sizes and chunks.
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty group.")

    root_ngff = _infer_ngff_version(root)
    root_zarr = 3 if root_ngff == "0.5" else 2

    if ngff_version is not None and ngff_version != root_ngff:
        raise ValueError(
            f"Cannot create a group with ngff_version={ngff_version} inside a root "
            f"dataset with ngff_version={root_ngff}."
        )

    if zarr_format is not None and zarr_format != root_zarr:
        raise ValueError(
            f"Cannot create a group with zarr_format={zarr_format} inside a root "
            f"dataset with zarr_format={root_zarr}."
        )

    requested_data_type = normalize_data_type(data_type or metadata.get("data_type"))
    label_group = requested_data_type == "label"

    cfg = ZarrWriteConfig(
        ngff_version=root_ngff,
        zarr_format=root_zarr,
        compressor=compressor,
        compressor_level=compressor_level,
        data_type=requested_data_type,
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    parent = root.require_group("labels") if label_group else root
    if group_name in parent:
        del parent[group_name]
    grp = parent.create_group(group_name)

    sizes = [tuple(s) for s in metadata["size"]]
    chunks = [tuple(c) for c in metadata["chunksize"]]
    dtype = metadata.get("dtype", "uint16")
    axes = normalize_axes(metadata.get("axes"), ndim=len(sizes[0]))

    effective_metadata = _normalize_metadata_for_write(
        metadata,
        axes,
        data_type=requested_data_type,
    )
    dummy_levels = [da.empty(shape=s, dtype=dtype, chunks=c) for s, c in zip(sizes, chunks)]
    _validate_metadata(dummy_levels, effective_metadata, axes)

    _write_empty_pyramid_arrays(
        grp,
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
        name=group_name,
        n_levels=len(dummy_levels),
    )
    _set_dimension_names(grp, multiscales["datasets"], axes, zarr_format=zarr_format)
    omero, extra = _ngff_payload_parts(
        dummy_levels[0],
        effective_metadata,
        axes,
        image_label_source="../../" if label_group else None,
    )
    if not label_group:
        extra = {"image-source": {"source": {"image": "../"}}}

    _set_group_ngff_metadata(
        grp,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )
    if label_group:
        _register_label_on_root(root, group_name, ngff_version)

    return grp
