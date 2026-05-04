from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from collections.abc import Sequence

import dask.array as da

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
        
    def reorder_channels(
        self, 
        new_order: List[int],
    ) -> None:
        """Reorder the channel axis and update channel-related metadata.

        Parameters
        ----------
            new_order : List[int]
                A permutation of the channel indices.
        """
        from .utils.metadata import reorder_channel_axis

        self.data, self.metadata = reorder_channel_axis(
            self.data,
            self.metadata,
            new_order,
            dataset_name="raw",
        )
        print(f"Channels reordered to {list(new_order)}")
        
    def update_metadata(
        self, 
        updates: Dict[str, Any]
    ) -> None:
        """Safely update entries in the metadata dictionary with validation.

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
        from .utils.metadata import apply_metadata_updates

        updated = apply_metadata_updates(
            self.data,
            self.metadata,
            updates,
            dataset_name="raw",
        )
        for key in updated:
            print(f"Updated metadata entry {key!r}")
            
    def subset_dataset(self,
                    T: Optional[Sequence[int]] = None,
                    C: Optional[Sequence[int]] = None,
                    Z: Optional[Sequence[int]] = None,
                    Y: Optional[Sequence[int]] = None,
                    X: Optional[Sequence[int]] = None
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
        self.build_pyramid(num_levels=num_levels, downscale_factor=downscale_factor)

        print("Dataset subset complete.")
