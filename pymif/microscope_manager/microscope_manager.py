from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional,Sequence
import dask.array as da
import warnings

from typing import TYPE_CHECKING, Optional
from collections.abc import Sequence

if TYPE_CHECKING:
    import napari

class MicroscopeManager(ABC):
    """
    Abstract base class for managing microscope image datasets.

    Provides shared functionality for reading, writing, visualizing,
    and managing multiscale image data and metadata.
    """
    
    def __init__(self):
        """Initialize the common manager state.

        Subclasses populate :attr:`data` with one dask array per pyramid level
        and :attr:`metadata` with the normalized PyMIF metadata schema.
        ``_open_files`` stores any file handles that should be closed through
        :meth:`close`.
        """
        self.data: List[da.Array] = []
        self.metadata: Dict[str, Any] = {}
        self._open_files = []

    @abstractmethod
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Abstract method that must be implemented by subclasses to read image data and metadata.

        Returns
        ----------
        Tuple
            - List of Dask arrays for each resolution level.
            - Dictionary of extracted metadata.
        """
        pass

    def to_zarr(self, 
                path: str,
                **kwargs) -> None:
        """Write the current dataset to an OME-Zarr store.

        Parameters
        ----------
        path : str
            Output zarr path.
        **kwargs
            Keyword arguments forwarded to :class:`~pymif.microscope_manager.utils.ngff.ZarrWriteConfig`,
            such as ``ngff_version``, ``zarr_format``, ``compressor``,
            ``compressor_level`` or ``overwrite``.
        """
        from .utils.to_zarr import to_zarr as _to_zarr
        from .utils.ngff import ZarrWriteConfig
        return _to_zarr(path, 
                      self.data, 
                      self.metadata, 
                      config=ZarrWriteConfig(**kwargs)
                      )

    def visualize(
        self,
        start_level: int = 0,
        stop_level: int = -1,
        in_memory: bool = False,
        viewer: "napari.Viewer | None" = None,
    ) -> Any:
        """Open the dataset in napari using the shared visualization helper.

        Parameters
        ----------
        start_level, stop_level : int
            Pyramid levels to expose. ``stop_level=-1`` means all available
            levels from ``start_level`` onward.
        in_memory : bool
            If ``True``, compute the selected levels before handing them to
            napari. Otherwise keep the dask-backed lazy representation.
        viewer : napari.Viewer | None
            Existing napari viewer to reuse. When omitted, a new viewer is
            created by the helper function.
        """
        from .utils.visualize import visualize as _visualize
        return _visualize(
            self.data,
            self.metadata,
            start_level=start_level,
            stop_level=stop_level,
            in_memory=in_memory,
            viewer=viewer,
        )

    def build_pyramid(self, 
                      num_levels: Optional[int] = 3, 
                      downscale_factor: int | Sequence[int] | None = 2,
                      start_level: Optional[int] = 0,
                      ) -> None:
        """Build additional pyramid levels from the current base-resolution data.

        The resulting data and scale metadata replace ``self.data`` and
        ``self.metadata`` in-place. This is mainly useful for managers that
        initially expose only one resolution level.
        """
        from .utils.pyramid import build_pyramid as _build_pyramid
        self.data, self.metadata = _build_pyramid(
            self.data, self.metadata, 
            num_levels=num_levels, 
            downscale_factor=2 if downscale_factor is None else downscale_factor,
            start_level = start_level,
        )

    def close(self) -> None:
        """Close all open resources, such as file handles."""
        for f in getattr(self, "_open_files", []):
            try:
                f.close()
            except Exception as e:
                print(f"Warning: failed to close file: {e}")
        self._open_files = []
        
    def reorder_channels(self, 
                         new_order: List[int]
                         ) -> None:
        """
        Reorder the channel axis and update channel-related metadata.

        Parameters
        ----------
            new_order : List[int]
                A permutation of the channel indices.
        """
        if not self.data:
            raise ValueError("No data loaded.")

        axes = self.metadata.get("axes", "").lower()
        if "c" not in axes:
            raise ValueError("Dataset has no channel axis to reorder.")

        c_dim = axes.index("c")
        original_c = self.data[0].shape[c_dim]
        if sorted(new_order) != list(range(original_c)):
            raise ValueError(f"new_order must be a permutation of 0..{original_c - 1}")

        reordered = []
        for level in self.data:
            slicer = [slice(None)] * level.ndim
            slicer[c_dim] = new_order
            reordered.append(level[tuple(slicer)])
        self.data = reordered

        if "channel_names" in self.metadata:
            self.metadata["channel_names"] = [self.metadata["channel_names"][i] for i in new_order]
        if "channel_colors" in self.metadata:
            self.metadata["channel_colors"] = [self.metadata["channel_colors"][i] for i in new_order]

        self.metadata["size"] = [tuple(level.shape) for level in self.data]
        self.metadata["chunksize"] = [tuple(level.chunksize) for level in self.data]
        print(f"Channels reordered to {new_order}")
        
    def update_metadata(self, 
                        updates: Dict[str, Any]
                        ) -> None:
        """
        Safely update entries in the metadata dictionary with validation.

        Parameters
        ----------
            updates : Dict[str, Any]
                Dictionary of key-value updates.

                Supports:
                    - channel_names (list[str])
                    - channel_colors (list[str]): valid matplotlib colors or hex code
                    - scales (list[tuple])
                    - time_increment (float)
                    - time_increment_unit (str)

        Warnings
        ----------
            Issues warnings or raises exceptions if updates are incompatible.
        """
        
        valid_keys = {
            "channel_names",
            "channel_colors",
            "scales",
            "time_increment",
            "time_increment_unit",
            "units",
            "data_type",
        }

        from matplotlib.colors import cnames
        import re

        HEX_PATTERN = re.compile(r'^#?[0-9a-fA-F]{6}$')

        def parse_color(v: str) -> str:
            """Parse a CLI color input:
            - Accept 6-digit hex codes (# optional)
            - Accept color names from matplotlib.colors.cnames
            - Raise a meaningful error if invalid
            """

            # --- 1) Hex code (with or without #) ---
            if HEX_PATTERN.match(v):
                return v.replace("#", "").upper()

            # --- 2) Matplotlib named color ---
            lower = v.lower()
            if lower in cnames:
                # cnames returns a hex string with '#', e.g. "#ff00ff"
                return cnames[lower].replace("#", "").upper()

            # --- 3) Fail: report detailed reason ---
            raise TypeError(
                f"Invalid color '{v}'. "
                f"Must be:\n"
                f"  • A 6-digit hex code (e.g. FF00FF or #ff00ff), OR\n"
                f"  • A valid color name from matplotlib ({', '.join(list(cnames.keys())[:10])}, ...)"
            )
        
        for key, value in updates.items():
            if key not in valid_keys:
                warnings.warn(f"⚠️ Unsupported or unknown metadata key: '{key}'")
                continue

            if key in {"channel_names", "channel_colors"}:
                axes = self.metadata.get("axes", "").lower()
                if "c" not in axes:
                    warnings.warn(f"Dataset has no channel axis. Skipping '{key}'.")
                    continue
                c_dim = axes.index("c")
                expected_len = self.data[0].shape[c_dim]
                if len(value) != expected_len:
                    warnings.warn(
                        f"Length of '{key}' ({len(value)}) does not match number of channels ({expected_len}). Skipping."
                    )
                    continue

            if key == "scales":
                if not isinstance(value, list) or len(value) != len(self.data):
                    raise ValueError("❌ 'scales' must be a list with one entry per pyramid level.")
                spatial_axes = [ax for ax in self.metadata.get("axes", "").lower() if ax in "zyx"]
                for s in value:
                    if not isinstance(s, (list, tuple)) or len(s) != len(spatial_axes):
                        raise ValueError("Each scale entry must match the dataset spatial axes.")

            if key == "time_increment":
                if not isinstance(value, (float, int)) or value <= 0:
                    raise ValueError("❌ 'time_increment' must be a positive float.")

            if key == "time_increment_unit":
                if value is not None and not isinstance(value, str):
                    raise ValueError("❌ 'time_increment_unit' must be a string or None.")

            if key == "units":
                if not isinstance(value, (tuple, list)):
                    raise TypeError("'units' must be a tuple or list.")
                spatial_axes = [ax for ax in self.metadata.get("axes", "").lower() if ax in "zyx"]
                if len(value) != len(spatial_axes):
                    raise ValueError("'units' must match the dataset spatial axes.")

            if key == "data_type":
                from .utils.axes import normalize_data_type
                value = normalize_data_type(value)
                self.metadata["is_label"] = value == "label"
                
            if key == "channel_colors":
                value = [parse_color(v) for v in value]

            self.metadata[key] = value
            print(f"✅ Updated metadata entry '{key}'")
            
    def subset_dataset(self,
                    T: Optional[Sequence[int]] = None,
                    C: Optional[Sequence[int]] = None,
                    Z: Optional[Sequence[int]] = None,
                    Y: Optional[Sequence[int]] = None,
                    X: Optional[Sequence[int]] = None,
                    rebuild_pyramid: bool = True
                    ) -> None:
        """
        Subset the dataset by timepoints, channels, or spatial coordinates.

        Parameters
        ----------
        T, C, Z, Y, X : Optional[Sequence[int]]
            Optional sequences of indices for each axis.
            Must be uniformly spaced. For example:
            dataset.subset_dataset(T=np.arange(0, 10, 2), Z=[0,1,2])

        Raises
        -------
        ValueError 
            if index spacing is not uniform or out of bounds.
        """
        from .utils.subset import subset_dask_array, subset_metadata
        from .utils.axes import index_list_from_selection
        import numpy as np
        
        if not self.data:
            raise ValueError("No data loaded.")

        shape = self.metadata["size"][0]
        axis_order = self.metadata["axes"].lower()
        requested = {"t": T, "c": C, "z": Z, "y": Y, "x": X}
        for name, index in requested.items():
            if index is None or name not in axis_order:
                continue
            axis = axis_order.index(name)
            indices = index_list_from_selection(index, shape[axis])
            if indices and (min(indices) < 0 or max(indices) >= shape[axis]):
                raise ValueError(f"Index for {name.upper()} out of range.")

        num_levels = len(self.data)
        downscale_factor = 2
        if num_levels > 1 and self.metadata["size"][1][-1] != 0:
            downscale_factor = int(np.round(self.metadata["size"][0][-1] / self.metadata["size"][1][-1]))

        subset_kwargs = {
            "T": T if "t" in axis_order else None,
            "C": C if "c" in axis_order else None,
            "Z": Z if "z" in axis_order else None,
            "Y": Y if "y" in axis_order else None,
            "X": X if "x" in axis_order else None,
        }
        self.data = [subset_dask_array(self.data[0], axes=axis_order, **subset_kwargs)]
        self.metadata = subset_metadata(self.metadata, **subset_kwargs)
        self.metadata["chunksize"] = [tuple(arr.chunksize) for arr in self.data]

        if rebuild_pyramid:
            self.build_pyramid(num_levels=num_levels, downscale_factor=downscale_factor)


        print("Dataset subset complete.")
