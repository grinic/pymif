from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import os

import dask.array as da
import zarr

from .microscope_manager import MicroscopeManager
from collections.abc import Iterator, Sequence

if TYPE_CHECKING:
    import napari

from dataclasses import dataclass

@dataclass
class ZarrDataset(Sequence):
    data: List[da.Array]
    zarr_data: List[Any] | None
    metadata: Dict[str, Any]
    name: str | None = None
    path: str | None = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self) -> Iterator[da.Array]:
        return iter(self.data)

    def __repr__(self):
        return (
            f"ZarrDataset("
            f"name={self.name!r}, "
            f"levels={len(self.data)}, "
            f"shape={self.data[0].shape if self.data else None}, "
            f"dtype={self.data[0].dtype if self.data else None}, "
            f"axes={self.metadata.get('axes')!r}"
            f")"
        )

class AttrDict(dict):
    """Dictionary with optional attribute access for valid Python names."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

class ZarrManager(MicroscopeManager):
    """
    A manager class for reading and handling OME-Zarr datasets.

    Supports both:
    - NGFF / OME-Zarr v0.4  -> metadata in root/group attrs directly
    - NGFF / OME-Zarr v0.5  -> metadata in attrs["ome"]
    """

    def __init__(
        self,
        path,
        chunks: Tuple[int, ...] = None,
        mode: str = "r",
        metadata: dict[str, Any] = None,
        ngff_version: str | None = None,
        zarr_format: int | None = None,
    ):
        """Open or create an OME-Zarr dataset.

        Parameters
        ----------
        path : str or path-like
            Path to the zarr root directory.
        chunks : tuple[int, ...] | None
            Optional dask chunking to use when lazily reopening arrays.
        mode : {"r", "r+", "a", "w"}
            Open mode. ``"r"`` reads an existing dataset, ``"a"`` opens or
            creates one, and ``"w"`` creates a new root from metadata.
        metadata : dict | None
            Metadata required when creating an empty dataset. Ignored when
            reading an existing dataset.
        ngff_version, zarr_format : optional
            Explicit format override used when creating a new store.
        """
        super().__init__()
        self.path = path
        self.chunks = chunks
        self.mode = mode
        self.metadata = metadata
        self.ngff_version: str | None = None
        self.ngff_version_override = ngff_version
        self.zarr_format_override = zarr_format

        if os.path.exists(self.path):
            if mode in ("r", "a", "r+"):
                self.root = zarr.open(zarr.storage.LocalStore(self.path), mode=self.mode)
                self.read()
            else:
                raise FileNotFoundError(
                    f"Zarr path {self.path} exists and mode='{mode}' is write-only. "
                    "Please select mode='r', 'r+' or 'a'."
                )
        else:
            if mode in ("w", "a"):
                from .utils.ngff import ZarrWriteConfig, _resolve_format

                cfg = ZarrWriteConfig(
                    ngff_version=self.ngff_version_override or (self.metadata or {}).get("ngff_version"),
                    zarr_format=self.zarr_format_override or (self.metadata or {}).get("zarr_format"),
                )
                resolved_ngff_version, resolved_zarr_format = _resolve_format(cfg)

                self.root = zarr.open_group(
                    zarr.storage.LocalStore(self.path),
                    mode=self.mode,
                    zarr_format=resolved_zarr_format,
                )

                from .utils.create_empty_dataset import create_empty_dataset as _create_empty_dataset
                _create_empty_dataset(
                    self.root,
                    self.metadata,
                    ngff_version=resolved_ngff_version,
                    zarr_format=resolved_zarr_format,
                )
            else:
                raise FileNotFoundError(
                    f"Zarr path {self.path} does not exist and mode='{mode}' is read-only. "
                    "Please select mode='w' or 'a'."
                )

    def _get_image_meta(self, group: zarr.Group) -> dict[str, Any]:
        """
        Return the image metadata dictionary for either NGFF v0.4 or v0.5.

        v0.5 -> group.attrs["ome"]
        v0.4 -> group.attrs
        """
        attrs = group.attrs.asdict()
        if "ome" in attrs and isinstance(attrs["ome"], dict):
            self.ngff_version = attrs["ome"].get("version", "0.5")
            return attrs["ome"]

        self.ngff_version = "0.4"
        return attrs

    def _get_multiscales(self, group: zarr.Group) -> list[dict[str, Any]]:
        """Return the NGFF ``multiscales`` block for ``group`` regardless of version."""
        image_meta = self._get_image_meta(group)
        return image_meta.get("multiscales", [])

    def _get_omero(self, group: zarr.Group) -> dict[str, Any]:
        """Return the OMERO-style channel metadata stored on ``group``."""
        image_meta = self._get_image_meta(group)
        return image_meta.get("omero", {})

    def _extract_metadata(
        self,
        data_levels: List[da.Array],
        datasets: list[dict[str, Any]],
        multiscales: dict[str, Any],
        omero: dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate NGFF metadata blocks into the normalized PyMIF metadata schema."""
        axes_info = multiscales.get("axes", [])
        axis_names = [a["name"] for a in axes_info]
        axes = "".join(axis_names)

        sizes = [tuple(arr.shape) for arr in data_levels]
        chunksize = [arr.chunksize for arr in data_levels]
        dtype = data_levels[0].dtype

        spatial_idx = [i for i, ax in enumerate(axis_names) if ax in ("z", "y", "x")]
        time_idx = axis_names.index("t") if "t" in axis_names else None
        channel_idx = axis_names.index("c") if "c" in axis_names else None

        scales = []
        for ds in datasets:
            ct = ds.get("coordinateTransformations", [{}])
            scale_vec = None
            if ct and isinstance(ct, list):
                scale_vec = ct[0].get("scale", None)

            if scale_vec is None:
                scales.append(tuple([1.0] * len(spatial_idx)))
            else:
                scales.append(tuple(scale_vec[i] for i in spatial_idx))

        units = tuple(axes_info[i].get("unit", None) for i in spatial_idx)

        if time_idx is not None:
            ct0 = datasets[0].get("coordinateTransformations", [{}])
            scale0 = None
            if ct0 and isinstance(ct0, list):
                scale0 = ct0[0].get("scale", None)

            time_increment = scale0[time_idx] if scale0 is not None else None
            time_increment_unit = axes_info[time_idx].get("unit", None)
        else:
            time_increment = None
            time_increment_unit = None

        channels = omero.get("channels", [])
        c_size = data_levels[0].shape[channel_idx] if channel_idx is not None else 1

        channel_names = [
            channels[i].get("label", f"Channel {i}") if i < len(channels) else f"Channel {i}"
            for i in range(c_size)
        ]
        channel_colors = [
            channels[i].get("color", "FFFFFF") if i < len(channels) else "FFFFFF"
            for i in range(c_size)
        ]

        return {
            "size": sizes,
            "chunksize": chunksize,
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_increment_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": str(dtype),
            "plane_files": None,
            "axes": axes,
            "ngff_version": self.ngff_version,
            "zarr_format": 3 if self.ngff_version == "0.5" else 2,
            }

    def _read_multiscale_group(
        self,
        group: zarr.Group,
    ) -> tuple[List[da.Array], List[Any], Dict[str, Any]]:
        """Load one multiscale image group as dask arrays plus normalized metadata."""
        multiscales_all = self._get_multiscales(group)
        if not multiscales_all:
            raise ValueError(f"Group '{group.name}' does not contain multiscales metadata.")

        multiscales = multiscales_all[0]
        datasets = multiscales.get("datasets", [])
        if not datasets:
            raise ValueError(f"Group '{group.name}' has multiscales metadata but no datasets.")

        omero = self._get_omero(group)

        data_levels = []
        zarr_levels = []

        for ds in datasets:
            path = ds["path"]
            zarr_array = group[path]
            zarr_levels.append(zarr_array)

            if self.chunks is not None and len(self.chunks) == len(zarr_array.shape):
                arr = da.from_zarr(zarr_array, chunks=self.chunks)
            else:
                arr = da.from_zarr(zarr_array)

            data_levels.append(arr)

        metadata = self._extract_metadata(
            data_levels=data_levels,
            datasets=datasets,
            multiscales=multiscales,
            omero=omero,
        )

        return data_levels, zarr_levels, metadata

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """Read the root image plus discover additional image groups and labels.

        The root image is exposed through ``self.data`` and ``self.metadata``.
        Additional multiscale subgroups are indexed in ``self.groups`` and label
        pyramids in ``self.labels``.
        """
        data_levels, zarr_levels, metadata = self._read_multiscale_group(self.root)

        self.raw = ZarrDataset(
            data=data_levels,
            zarr_data=zarr_levels,
            metadata=metadata,
            name="raw",
            path="/",
        )

        # Backward-compatible aliases
        self._sync_raw_aliases()
        self.chunks = data_levels[0].chunksize

        self.groups = AttrDict()
        self.labels = AttrDict()

        for name in self.root.group_keys():
            if name == "labels":
                self.labels = self._load_labels()
            else:
                group_dataset = self._load_group(name)

                if group_dataset is not None:
                    self.groups[name] = group_dataset

                    # Optional convenience: d.proc.data
                    if name.isidentifier() and not hasattr(self, name):
                        setattr(self, name, group_dataset)

        print(self.root.tree())
        for k, v in self.metadata.items():
            print(f"{k.upper()}: {v}")

        return self.data, self.metadata

    def _sync_raw_aliases(self) -> None:
        """
        Keep the old single-dataset API synchronized with the raw dataset.

        This preserves:
            d.data
            d.metadata
            d.zarr_data

        as aliases to:
            d.raw.data
            d.raw.metadata
            d.raw.zarr_data
        """
        self.data = self.raw.data
        self.metadata = self.raw.metadata
        self.zarr_data = self.raw.zarr_data

    def _iter_datasets(
        self,
        include_raw: bool = True,
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Iterate over all datasets managed by ZarrManager.

        This is ZarrManager-specific because only ZarrManager knows about
        raw/groups/labels.
        """
        if include_raw and hasattr(self, "raw"):
            yield "raw", self.raw

        if include_groups and hasattr(self, "groups"):
            for name, dataset in self.groups.items():
                yield name, dataset

        if include_labels and hasattr(self, "labels"):
            for name, dataset in self.labels.items():
                yield f"labels/{name}", dataset

    def _invalidate_zarr_data(self, dataset: ZarrDataset) -> None:
        """
        After lazy in-memory transformations, dataset.data no longer necessarily
        corresponds one-to-one to the original on-disk Zarr arrays.
        """
        dataset.zarr_data = None

    def _load_group(self, name):
        group = self.root[name]

        try:
            arrays, zarr_arrays, metadata = self._read_multiscale_group(group)
        except ValueError:
            return None

        metadata["is_label"] = False
        metadata["name"] = name

        return ZarrDataset(
            data=arrays,
            zarr_data=zarr_arrays,
            metadata=metadata,
            name=name,
            path=name,
        )

    def _load_labels(self):
        labels = AttrDict()

        if "labels" not in self.root:
            return labels

        labels_grp = self.root["labels"]

        for label_name, label_grp in labels_grp.groups():
            try:
                arrays, zarr_arrays, metadata = self._read_multiscale_group(label_grp)
            except ValueError:
                continue

            metadata = dict(metadata)
            metadata["is_label"] = True
            metadata["name"] = label_name

            labels[label_name] = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=metadata,
                name=label_name,
                path=f"labels/{label_name}",
            )

        return labels

    def visualize_zarr(
        self,
        viewer: "napari.Viewer | None " = None,
    ) -> "napari.Viewer | None":
        """Open the root image and all multiscale subgroups in napari."""
        try:
            import napari
        except ImportError:
            import warnings
            warnings.warn(
                "napari is not installed. Install with `pip install pymif[napari]` to use visualization.",
                stacklevel=2,
            )
            return None

        if viewer is None:
            viewer = napari.Viewer()

        def discover_multiscales(group: zarr.Group, path=""):
            results = []

            try:
                multiscales = self._get_multiscales(group)
                if multiscales:
                    results.append(path)
            except Exception:
                pass

            for name, sub in group.groups():
                if name != "labels":
                    subpath = f"{path}/{name}" if path else name
                    results.extend(discover_multiscales(sub, path=subpath))

            return results

        multiscale_paths = discover_multiscales(self.root)

        for gpath in multiscale_paths:
            full_path = Path(self.path) / gpath
            viewer.open(full_path, plugin="napari-ome-zarr")

        return viewer

    def create_empty_group(
        self,
        group_name: str,
        metadata: Dict[str, Any],
        is_label: bool = False,
    ):
        """Create an empty image subgroup or label subgroup and update this manager."""
        from .utils.create_empty_group import create_empty_group as _create_empty_group

        grp = _create_empty_group(
            root=self.root,
            group_name=group_name,
            metadata=metadata,
            is_label=is_label,
        )

        # Read the newly created group back into the in-memory ZarrDataset model.
        if is_label:
            arrays, zarr_arrays, group_metadata = self._read_multiscale_group(grp)

            group_metadata = dict(group_metadata)
            group_metadata["is_label"] = True
            group_metadata["name"] = group_name

            dataset = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=group_metadata,
                name=group_name,
                path=f"labels/{group_name}",
            )

            if not hasattr(self, "labels"):
                self.labels = AttrDict()

            self.labels[group_name] = dataset

        else:
            arrays, zarr_arrays, group_metadata = self._read_multiscale_group(grp)

            group_metadata = dict(group_metadata)
            group_metadata["is_label"] = False
            group_metadata["name"] = group_name

            dataset = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=group_metadata,
                name=group_name,
                path=group_name,
            )

            if not hasattr(self, "groups"):
                self.groups = AttrDict()

            self.groups[group_name] = dataset

            # Optional convenience: d.proc.data
            if group_name.isidentifier():
                setattr(self, group_name, dataset)

        return dataset

    def write_image_region(
        self,
        data,
        t: int | slice = slice(None),
        c: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: Optional[str] = None,
        downscale_factor: int | Sequence[int] | None = None,
    ):
        """Write an image patch into a root or subgroup pyramid and refresh lower levels."""
        from .utils.write_image_region import write_image_region as _write_image_region
        return _write_image_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t,
            c=c,
            z=z,
            y=y,
            x=x,
            level=level,
            group_name=group,
            downscale_factor=downscale_factor,
        )

    def write_label_region(
        self,
        data,
        t: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: str = None,
        downscale_factor: int | Sequence[int] | None = None,
    ):
        """Write a label patch into a label pyramid and regenerate coarser levels."""
        from .utils.write_label_region import write_label_region as _write_label_region
        return _write_label_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t,
            z=z,
            y=y,
            x=x,
            level=level,
            group_name=group,
            downscale_factor=downscale_factor,
        )
    
    def subset_dataset(
        self,
        T=None,
        C=None,
        Z=None,
        Y=None,
        X=None,
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Subset raw, groups, and labels.

        Channel subsetting is skipped automatically for datasets without a
        channel axis, such as most label datasets.
        """
        from .utils.subset import subset_dask_array, subset_metadata
        from .utils.pyramid import build_pyramid as _build_pyramid
        import numpy as np

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.data:
                continue

            axes = dataset.metadata.get("axes", "").lower()

            subset_kwargs = {
                "T": T if "t" in axes else None,
                "C": C if "c" in axes else None,
                "Z": Z if "z" in axes else None,
                "Y": Y if "y" in axes else None,
                "X": X if "x" in axes else None,
            }

            # Validate requested indices against this specific dataset.
            shape = dataset.data[0].shape
            for ax_name, user_index in {
                "t": subset_kwargs["T"],
                "c": subset_kwargs["C"],
                "z": subset_kwargs["Z"],
                "y": subset_kwargs["Y"],
                "x": subset_kwargs["X"],
            }.items():
                if user_index is None or ax_name not in axes:
                    continue

                axis = axes.index(ax_name)
                max_size = shape[axis]

                indices = np.atleast_1d(user_index)
                if indices.size > 0:
                    if indices.min() < 0 or indices.max() >= max_size:
                        raise ValueError(
                            f"Index for axis '{ax_name}' out of range in dataset "
                            f"{name!r}. Axis size is {max_size}."
                        )

            num_levels = len(dataset.data)

            dataset.data = [
                subset_dask_array(
                    dataset.data[0],
                    axes=axes,
                    **subset_kwargs,
                )
            ]

            dataset.metadata = subset_metadata(
                dataset.metadata,
                **subset_kwargs,
            )

            # Rebuild pyramid to preserve the number of levels.
            dataset.data, dataset.metadata = _build_pyramid(
                dataset.data,
                dataset.metadata,
                num_levels=num_levels,
            )

            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        print("Zarr datasets subset complete.")

    def build_pyramid(
        self,
        num_levels: Optional[int] = 3,
        downscale_factor: int | Sequence[int] | None = 2,
        start_level: int = 0,
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Build/rebuild pyramids for raw, groups, and labels.
        """
        from .utils.pyramid import build_pyramid as _build_pyramid

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.data:
                continue

            dataset.data, dataset.metadata = _build_pyramid(
                dataset.data,
                dataset.metadata,
                num_levels=num_levels,
                downscale_factor=downscale_factor,
                start_level=start_level,
            )

            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        print("Zarr pyramids rebuilt.")

    def reorder_channels(
        self,
        new_order: List[int],
        include_groups: bool = True,
    ):
        """
        Reorder channels in raw and non-label image groups.

        Labels are skipped because they usually do not have a channel axis.
        """
        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=False,
        ):
            if not dataset.data:
                continue

            axes = dataset.metadata.get("axes", "").lower()

            if "c" not in axes:
                continue

            c_axis = axes.index("c")
            n_channels = dataset.data[0].shape[c_axis]

            if sorted(new_order) != list(range(n_channels)):
                raise ValueError(
                    f"new_order must be a permutation of 0..{n_channels - 1} "
                    f"for dataset {name!r}."
                )

            reordered_levels = []

            for level in dataset.data:
                slicer = [slice(None)] * level.ndim
                slicer[c_axis] = new_order
                reordered_levels.append(level[tuple(slicer)])

            dataset.data = reordered_levels

            if "channel_names" in dataset.metadata:
                dataset.metadata["channel_names"] = [
                    dataset.metadata["channel_names"][i]
                    for i in new_order
                ]

            if "channel_colors" in dataset.metadata:
                dataset.metadata["channel_colors"] = [
                    dataset.metadata["channel_colors"][i]
                    for i in new_order
                ]

            dataset.metadata["size"] = [tuple(arr.shape) for arr in dataset.data]
            dataset.metadata["chunksize"] = [arr.chunksize for arr in dataset.data]

            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        print(f"Channels reordered to {new_order}.")     

    def update_metadata(
        self,
        updates: Dict[str, Any],
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Update metadata for raw, groups, and labels.

        Channel-specific metadata is skipped for datasets without a channel axis.
        """
        import re
        import warnings
        from matplotlib.colors import cnames

        valid_keys = {
            "channel_names",
            "channel_colors",
            "scales",
            "time_increment",
            "time_increment_unit",
            "units",
        }

        hex_pattern = re.compile(r"^#?[0-9a-fA-F]{6}$")

        def parse_color(value: str) -> str:
            if not isinstance(value, str):
                raise TypeError("Channel colors must be strings.")

            if hex_pattern.match(value):
                return value.replace("#", "").upper()

            lower = value.lower()
            if lower in cnames:
                return cnames[lower].replace("#", "").upper()

            raise TypeError(
                f"Invalid color {value!r}. Use a 6-digit hex code or a valid "
                "matplotlib color name."
            )

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.metadata:
                continue

            axes = dataset.metadata.get("axes", "").lower()

            for key, value in updates.items():
                if key not in valid_keys:
                    warnings.warn(
                        f"Unsupported or unknown metadata key {key!r}.",
                        stacklevel=2,
                    )
                    continue

                if key in {"channel_names", "channel_colors"}:
                    if "c" not in axes:
                        continue

                    c_axis = axes.index("c")
                    expected_channels = dataset.data[0].shape[c_axis]

                    if len(value) != expected_channels:
                        warnings.warn(
                            f"Skipping {key!r} for dataset {name!r}: expected "
                            f"{expected_channels} values, got {len(value)}.",
                            stacklevel=2,
                        )
                        continue

                    if key == "channel_colors":
                        value = [parse_color(v) for v in value]

                elif key == "scales":
                    if not isinstance(value, list):
                        raise TypeError("'scales' must be a list.")

                    if len(value) != len(dataset.data):
                        raise ValueError(
                            f"'scales' must contain one entry per pyramid level "
                            f"for dataset {name!r}. Expected {len(dataset.data)}, "
                            f"got {len(value)}."
                        )

                    for scale in value:
                        if not isinstance(scale, (tuple, list)):
                            raise TypeError(
                                "Each scale entry must be a tuple or list."
                            )

                elif key == "time_increment":
                    if value is not None and (
                        not isinstance(value, (int, float)) or value <= 0
                    ):
                        raise ValueError(
                            "'time_increment' must be a positive number or None."
                        )

                elif key == "time_increment_unit":
                    if value is not None and not isinstance(value, str):
                        raise TypeError(
                            "'time_increment_unit' must be a string or None."
                        )

                elif key == "units":
                    if not isinstance(value, (tuple, list)):
                        raise TypeError("'units' must be a tuple or list.")

                dataset.metadata[key] = value

        self._sync_raw_aliases()

        print("Zarr metadata updated.")

    def to_zarr(
        self,
        path: str | Path,
        include_groups: bool = True,
        include_labels: bool = True,
        reread: bool = True,
        **kwargs,
    ):
        """Write raw data, groups, and labels to a complete OME-Zarr store.

        Raw data is written directly to the root group.
        Image groups are written as root-level subgroups.
        Labels are written under /labels.
        """
        from .utils.to_zarr import write_multiscale_to_group
        from .utils.ngff import ZarrWriteConfig, _resolve_format

        cfg = ZarrWriteConfig(**kwargs)
        ngff_version, zarr_format = _resolve_format(cfg)

        root = zarr.open_group(
            str(Path(path)),
            mode="w" if cfg.overwrite else "w-",
            zarr_format=zarr_format,
        )

        # ------------------------------------------------------------
        # 1. Raw dataset at root
        # ------------------------------------------------------------
        write_multiscale_to_group(
            group=root,
            data_levels=self.raw.data,
            metadata=self.raw.metadata,
            config=cfg,
            name=self.raw.name or "raw",
            is_label=False,
        )

        # ------------------------------------------------------------
        # 2. Non-label image groups at root/group_name
        # ------------------------------------------------------------
        if include_groups:
            for group_name, dataset in self.groups.items():
                group = root.require_group(group_name)

                write_multiscale_to_group(
                    group=group,
                    data_levels=dataset.data,
                    metadata=dataset.metadata,
                    config=cfg,
                    name=dataset.name or group_name,
                    is_label=False,
                )

        # ------------------------------------------------------------
        # 3. Label groups at root/labels/label_name
        # ------------------------------------------------------------
        if include_labels and self.labels:
            labels_group = root.require_group("labels")

            label_names = []

            for label_name, dataset in self.labels.items():
                label_names.append(label_name)

                label_group = labels_group.require_group(label_name)

                write_multiscale_to_group(
                    group=label_group,
                    data_levels=dataset.data,
                    metadata=dataset.metadata,
                    config=cfg,
                    name=dataset.name or label_name,
                    is_label=True,
                )

            # Minimal root-level labels discovery metadata.
            if ngff_version == "0.5":
                ome = root.attrs.get("ome", {})
                ome["version"] = "0.5"
                ome["labels"] = [
                    {
                        "name": name,
                        "source": {
                            "image": "../",
                        },
                    }
                    for name in label_names
                ]
                root.attrs["ome"] = ome
            else:
                root.attrs["labels"] = [
                    {
                        "name": name,
                        "source": {
                            "image": "../",
                        },
                    }
                    for name in label_names
                ]

        self.path = str(path)
        self.root = root
        self.mode = "r+"
        self.ngff_version = ngff_version
        self.zarr_format_override = zarr_format

        if reread and cfg.compute:
            self.read()

        return root
