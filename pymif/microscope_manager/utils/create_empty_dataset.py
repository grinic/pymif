from typing import Dict, Any
import zarr

from .ngff import (
    _set_group_ngff_metadata,
    _build_v2_compressor,
    _build_v3_compressors,
    ZarrWriteConfig,
    _resolve_format,
    _build_axes,
    _build_coordinate_transformations,
    _build_omero_metadata,
)

def create_empty_dataset(
    root: zarr.Group,
    metadata: Dict[str, Any],
    ngff_version: str | None = None,
    zarr_format: int | None = None,
    compressor: str | None = None,
    compressor_level: int = 3,
):
    if not metadata:
        raise ValueError("Metadata is required to create an empty dataset.")

    cfg = ZarrWriteConfig(
        ngff_version=ngff_version or metadata.get("ngff_version") or "0.5",
        zarr_format=zarr_format or metadata.get("zarr_format"),
        compressor=compressor,
        compressor_level=compressor_level,
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    sizes = metadata["size"]
    chunks = metadata["chunksize"]
    dtype = metadata.get("dtype", "uint16")
    axes = tuple(metadata["axes"])

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
        "type": "image",
    }

    import dask.array as da
    dummy = da.empty(shape=sizes[0], dtype=dtype, chunks=chunks[0])
    omero = _build_omero_metadata(dummy, axes, metadata)

    _set_group_ngff_metadata(
        root,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
    )