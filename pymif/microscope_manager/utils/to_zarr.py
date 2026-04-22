from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import dask.array as da
import numpy as np
import zarr

from .ngff import ( _build_axes, _build_coordinate_transformations, 
                   _build_omero_metadata, _validate_metadata, _resolve_format, 
                   _write_pyramid_v3, _write_pyramid_v2, ZarrWriteConfig
)

DEFAULT_COLORS = (
    "FF0000", "00FF00", "0000FF", "FFFF00",
    "FF00FF", "00FFFF", "FFFFFF", "808080",
)
SPATIAL_AXES = {"z", "y", "x"}


def to_zarr(
    path: str | Path,
    data_levels: Sequence[da.Array],
    metadata: dict[str, Any],
    *,
    config: ZarrWriteConfig | None = None,
):
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("`data_levels` cannot be empty.")

    axes = tuple(metadata["axes"])
    _validate_metadata(data_levels, metadata, axes)

    ngff_version, zarr_format = _resolve_format(cfg)

    root = zarr.open_group(
        str(Path(path)),
        mode="w" if cfg.overwrite else "w-",
        zarr_format=zarr_format,
    )

    if zarr_format == 3:
        delayed = _write_pyramid_v3(
            root=root,
            data_levels=data_levels,
            cfg=cfg,
        )
    else:
        delayed = _write_pyramid_v2(
            root=root,
            data_levels=data_levels,
            cfg=cfg,
        )

    multiscales = {
        "name": metadata.get("name") or "dataset",
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
    }

    omero = _build_omero_metadata(data_levels[0], axes, metadata)

    if ngff_version == "0.5":
        root.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [multiscales],
            "omero": omero,
        }
    else:
        root.attrs["multiscales"] = [multiscales]
        root.attrs["omero"] = omero

    return root if cfg.compute else delayed
