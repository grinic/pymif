import dask.array as da
import zarr
from typing import List, Dict, Any

from .to_zarr import (
    ZarrWriteConfig,
    _resolve_format,
    _build_coordinate_transformations,
    _get_chunks,
    _build_v2_compressor,
    _build_v3_compressors,
)

from .ngff import _infer_ngff_version
from .ngff import _register_label_on_root

LABEL_AXES = ("t", "z", "y", "x")


def add_label(
    root: zarr.Group,
    mode: str,
    label_levels: List[da.Array],
    label_name: str,
    metadata: Dict[str, Any],
    compressor=None,
    compressor_level=3,
    parallelize=False,  # keep for API compatibility, but ignore
    ngff_version: str | None = None,
):
    if mode not in ("r+", "a", "w"):
        raise PermissionError(
            f"Dataset opened in read-only mode ('{mode}'). "
            "Reopen with mode='r+' to allow modifications."
        )

    if not label_levels:
        raise ValueError("`label_levels` cannot be empty.")

    expected_layers = len(metadata["size"])
    if len(label_levels) != expected_layers:
        raise ValueError(
            f"Label pyramid has {len(label_levels)} levels, expected {expected_layers}."
        )

    for i, level in enumerate(label_levels):
        expected_shape = (
            metadata["size"][i][0],  # t
            metadata["size"][i][2],  # z
            metadata["size"][i][3],  # y
            metadata["size"][i][4],  # x
        )
        if level.ndim != 4:
            raise ValueError(
                f"Label pyramid level {i} has {level.ndim} dimensions, expected 4 (tzyx)."
            )
        if tuple(level.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch at level {i}. Label shape={level.shape}, expected={expected_shape}."
            )

    inferred_ngff = _infer_ngff_version(root)
    cfg = ZarrWriteConfig(
        ngff_version=ngff_version or inferred_ngff,
        compressor=compressor,
        compressor_level=compressor_level,
        compute=True,
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    labels_grp = root.require_group("labels")
    if label_name in labels_grp:
        del labels_grp[label_name]
    label_grp = labels_grp.create_group(label_name)

    for i, arr in enumerate(label_levels):
        kwargs = {
            "name": str(i),
            "shape": arr.shape,
            "dtype": arr.dtype,
            "chunks": _get_chunks(arr),
        }
        if zarr_format == 2:
            kwargs["compressor"] = _build_v2_compressor(cfg.compressor, cfg.compressor_level)
        else:
            compressors = _build_v3_compressors(cfg.compressor, cfg.compressor_level)
            if compressors is not None:
                kwargs["compressors"] = compressors

        z = label_grp.create_array(**kwargs)
        if isinstance(arr, da.Array):
            da.store(arr, z, lock=False, compute=True)
        else:
            z[:] = arr

    coordinate_transformations = _build_coordinate_transformations(
        axes=LABEL_AXES,
        scales=metadata["scales"],
        time_increment=metadata.get("time_increment"),
    )

    label_multiscales = {
        "name": label_name,
        "axes": [
            {"name": "t", "type": "time", "unit": metadata.get("time_increment_unit")},
            {"name": "z", "type": "space", "unit": metadata["units"][0]},
            {"name": "y", "type": "space", "unit": metadata["units"][1]},
            {"name": "x", "type": "space", "unit": metadata["units"][2]},
        ],
        "datasets": [
            {"path": str(i), "coordinateTransformations": coordinate_transformations[i]}
            for i in range(len(label_levels))
        ],
    }

    _set_group_ngff_metadata(
        label_grp,
        ngff_version=ngff_version,
        multiscales=label_multiscales,
        omero=None,
        extra={"image-label": {"source": {"image": "../../"}}},
    )

    _register_label_on_root(root, label_name, ngff_version)





