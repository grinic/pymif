from typing import Dict, Any
import zarr
import dask.array as da

from .to_zarr import (
    ZarrWriteConfig,
    _resolve_format,
    _build_axes,
    _build_coordinate_transformations,
    _build_omero_metadata,
    _build_v2_compressor,
    _build_v3_compressors,
)

from .ngff import _set_group_ngff_metadata

def create_empty_group(
    root: zarr.Group,
    group_name: str,
    metadata: Dict[str, Any],
    is_label: bool = False,
    ngff_version: str | None = None,
    compressor: str | None = None,
    compressor_level: int = 3,
):
    if not metadata:
        raise ValueError("Metadata is required to create an empty group.")

    inferred_ngff = _infer_ngff_version(root)
    cfg = ZarrWriteConfig(
        ngff_version=ngff_version or inferred_ngff,
        compressor=compressor,
        compressor_level=compressor_level,
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    if is_label:
        parent = root.require_group("labels")
    else:
        parent = root

    if group_name in parent:
        del parent[group_name]
    grp = parent.create_group(group_name)

    sizes = metadata["size"]
    chunks = metadata["chunksize"]
    dtype = metadata.get("dtype", "uint16")

    if is_label:
        axes = ("t", "z", "y", "x")
        level_shapes = [(s[0], s[2], s[3], s[4]) for s in sizes]
        level_chunks = [(c[0], c[2], c[3], c[4]) for c in chunks]
    else:
        axes = tuple(metadata["axes"])
        level_shapes = sizes
        level_chunks = chunks

    for i, (shape, chunk) in enumerate(zip(level_shapes, level_chunks)):
        kwargs = {
            "name": str(i),
            "shape": shape,
            "chunks": chunk,
            "dtype": dtype,
        }
        if zarr_format == 2:
            kwargs["compressor"] = _build_v2_compressor(compressor, compressor_level)
        else:
            compressors = _build_v3_compressors(compressor, compressor_level)
            if compressors is not None:
                kwargs["compressors"] = compressors
        grp.create_array(**kwargs)

    if is_label:
        axes_meta = [
            {"name": "t", "type": "time", "unit": metadata.get("time_increment_unit")},
            {"name": "z", "type": "space", "unit": metadata["units"][0]},
            {"name": "y", "type": "space", "unit": metadata["units"][1]},
            {"name": "x", "type": "space", "unit": metadata["units"][2]},
        ]
    else:
        axes_meta = _build_axes(tuple(metadata["axes"]), metadata)

    coordinate_transformations = _build_coordinate_transformations(
        axes=axes,
        scales=metadata["scales"],
        time_increment=metadata.get("time_increment"),
    )

    multiscales = {
        "name": group_name,
        "axes": axes_meta,
        "datasets": [
            {"path": str(i), "coordinateTransformations": coordinate_transformations[i]}
            for i in range(len(level_shapes))
        ],
    }

    if is_label:
        _set_group_ngff_metadata(
            grp,
            ngff_version=ngff_version,
            multiscales=multiscales,
            omero=None,
            extra={"image-label": {"source": {"image": "../../"}}},
        )
        _register_label_on_root(root, group_name, ngff_version)
    else:
        dummy = da.empty(shape=level_shapes[0], dtype=dtype, chunks=level_chunks[0])
        omero = _build_omero_metadata(dummy, tuple(metadata["axes"]), metadata)
        _set_group_ngff_metadata(
            grp,
            ngff_version=ngff_version,
            multiscales=multiscales,
            omero=omero,
            extra={"image-source": {"source": {"image": "../"}}},
        )

    return grp