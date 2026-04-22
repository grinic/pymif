import zarr
import numpy as np
import dask.array as da
from typing import Any, Literal, Sequence

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