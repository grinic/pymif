from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import os

import dask.array as da
import zarr

from .microscope_manager import MicroscopeManager

if TYPE_CHECKING:
    import napari


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

            if self.chunks is None:
                arr = da.from_zarr(zarr_array)
            else:
                arr = da.from_zarr(zarr_array, chunks=self.chunks)

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

        self.data = data_levels
        self.zarr_data = zarr_levels
        self.metadata = metadata
        self.chunks = data_levels[0].chunksize

        self.groups = {}
        self.labels = {}

        for name in self.root.group_keys():
            if name == "labels":
                self.labels = self._load_labels()
            else:
                self.groups[name] = self._load_group(name)

        print(self.root.tree())
        for k, v in self.metadata.items():
            print(f"{k.upper()}: {v}")

        return self.data, self.metadata

    def _load_group(self, name):
        """Attempt to load a named subgroup if it contains NGFF image metadata."""
        group = self.root[name]
        try:
            arrays, _, _ = self._read_multiscale_group(group)
            return arrays
        except ValueError:
            return None

    def _load_labels(self) -> Dict[str, List[da.Array]]:
        """Discover label pyramids stored below the root ``labels`` group."""
        labels: Dict[str, List[da.Array]] = {}

        if "labels" not in self.root:
            return labels

        labels_grp = self.root["labels"]

        for label_name, label_grp in labels_grp.groups():
            multiscales_all = self._get_multiscales(label_grp)
            if not multiscales_all:
                continue

            multiscales = multiscales_all[0]
            datasets = multiscales.get("datasets", [])
            if not datasets:
                continue

            arrays = []
            for ds in datasets:
                arr_path = ds["path"]
                zarr_array = label_grp[arr_path]

                if self.chunks is not None and len(self.chunks) == len(zarr_array.shape):
                    arr = da.from_zarr(zarr_array, chunks=self.chunks)
                else:
                    arr = da.from_zarr(zarr_array)

                arrays.append(arr)

            labels[label_name] = arrays

        return labels

    def add_label(
        self,
        label_levels: List[da.Array],
        label_name: str = "new_label",
        compressor: Any = None,
        compressor_level: Any = 3,
        parallelize: Any = False,
    ) -> None:
        """Register a new multiscale label pyramid in the current zarr store."""
        from .utils.add_label import add_label as _add_label
        return _add_label(
            root=self.root,
            mode=self.mode,
            label_levels=label_levels,
            label_name=label_name,
            metadata=self.metadata,
            compressor=compressor,
            compressor_level=compressor_level,
            parallelize=parallelize,
        )

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
        """Create an empty image subgroup or label subgroup below the current root."""
        from .utils.create_empty_group import create_empty_group as _create_empty_group
        return _create_empty_group(
            root=self.root,
            group_name=group_name,
            metadata=metadata,
            is_label=is_label,
        )

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
        )