from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional,Sequence
import dask.array as da
import napari
import warnings

class MicroscopeManager(ABC):
    """
    Abstract base class for managing microscope image datasets.

    Provides shared functionality for reading, writing, visualizing,
    and managing multiscale image data and metadata.
    """
    
    def __init__(self):
        self.data: List[da.Array] = []
        self.metadata: Dict[str, Any] = {}
        self._open_files = []

    @abstractmethod
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Abstract method that must be implemented by subclasses to read image data and metadata.

        Returns:
            Tuple containing:
                - List of Dask arrays for each resolution level.
                - Dictionary of extracted metadata.
        """
        pass

    def write(self, path: str, 
              compressor: Any = None, 
              compressor_level: Any = 3, 
              overwrite=True,
              parallelize: Any = False) -> None:
        from .utils.write import write as _write
        return _write(path, 
                      self.data, 
                      self.metadata, 
                      compressor=compressor, 
                      compressor_level=compressor_level,
                      overwrite=overwrite,
                      parallelize=parallelize
                      )

    def visualize(  self,
                    start_level: Optional[int] = 0,
                    stop_level: Optional[int] = -1,
                    in_memory: Optional[bool] = False,
                    viewer: Optional[napari.Viewer] = None,
                  ) -> Any:
        from .utils.visualize import visualize as _visualize
        return _visualize(  self.data, 
                            self.metadata,
                            start_level = start_level,
                            stop_level = stop_level,
                            in_memory = in_memory,
                            viewer = viewer,
                          )

    def build_pyramid(self, 
                      num_levels: Optional[int] = 3, 
                      downscale_factor: Optional[int] = 2,
                      start_level: Optional[int] = 0,
                      ) -> None:
        from .utils.pyramid import build_pyramid as _build_pyramid
        """
        Converts single-resolution data into a pyramidal multiscale structure
        and updates self.data and self.metadata in-place.
        """
        self.data, self.metadata = _build_pyramid(
            self.data, self.metadata, 
            num_levels=num_levels, 
            downscale_factor=downscale_factor,
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
        
    # def print_info(self) -> None:
    
    def reorder_channels(self, new_order: List[int]):
        """
        Reorder the channel axis and update channel-related metadata.

        Args:
            new_order (List[int]): A permutation of the channel indices.
        """
        if not self.data:
            raise ValueError("No data loaded.")

        c_dim = self.metadata["axes"].index("c")
        original_c = self.data[0].shape[c_dim]
        
        if sorted(new_order) != list(range(original_c)):
            raise ValueError(f"new_order must be a permutation of 0..{original_c - 1}")

        # Reorder each resolution level
        self.data = [
            da.moveaxis(level, c_dim, 1)[:, new_order, ...]
            for level in self.data
        ]

        # Reorder metadata
        if "channel_names" in self.metadata:
            self.metadata["channel_names"] = [self.metadata["channel_names"][i] for i in new_order]
        if "channel_colors" in self.metadata:
            self.metadata["channel_colors"] = [self.metadata["channel_colors"][i] for i in new_order]

        print(f"✅ Channels reordered to {new_order}")
        
    def update_metadata(self, updates: Dict[str, Any]):
        """
        Safely update entries in the metadata dictionary with validation.

        Args:
            updates (Dict[str, Any]): Dictionary of key-value updates.

        Supports:
            - channel_names (list[str])
            - channel_colors (list[int or str])
            - scales (list[tuple])
            - time_increment (float)
            - time_increment_unit (str)

        Warnings:
            Issues warnings or raises exceptions if updates are incompatible.
        """
        valid_keys = {
            "channel_names",
            "channel_colors",
            "scales",
            "time_increment",
            "time_increment_unit",
        }

        for key, value in updates.items():
            if key not in valid_keys:
                warnings.warn(f"⚠️ Unsupported or unknown metadata key: '{key}'")
                continue

            if key in {"channel_names", "channel_colors"}:
                c_dim = self.metadata["axes"].index("c")
                expected_len = self.data[0].shape[c_dim]
                if len(value) != expected_len:
                    warnings.warn(
                        f"⚠️ Length of '{key}' ({len(value)}) does not match number of channels ({expected_len}). Skipping."
                    )
                    continue

            if key == "scales":
                if not isinstance(value, list) or len(value) != len(self.data):
                    raise ValueError("❌ 'scales' must be a list with one entry per pyramid level.")
                for s in value:
                    if not isinstance(s, (list, tuple)) or len(s) != 3:
                        raise ValueError("❌ Each scale entry must be a tuple/list of (Z, Y, X).")

            if key == "time_increment":
                if not isinstance(value, (float, int)) or value <= 0:
                    raise ValueError("❌ 'time_increment' must be a positive float.")

            if key == "time_increment_unit":
                if not isinstance(value, str):
                    raise ValueError("❌ 'time_increment_unit' must be a string.")

            self.metadata[key] = value
            print(f"✅ Updated metadata entry '{key}'")
            
            
    def subset_dataset(self,
                    T: Optional[Sequence[int]] = None,
                    C: Optional[Sequence[int]] = None,
                    Z: Optional[Sequence[int]] = None,
                    Y: Optional[Sequence[int]] = None,
                    X: Optional[Sequence[int]] = None):
        """
        Subset the dataset by timepoints, channels, or spatial coordinates.

        Args:
            T, C, Z, Y, X: Optional sequences of indices for each axis.
                        Must be uniformly spaced. For example:
                        dataset.subset_dataset(T=np.arange(0, 10, 2), Z=[0,1,2])
        Raises:
            ValueError: if index spacing is not uniform or out of bounds.
        """
        from .utils.subset import subset_dask_array, subset_metadata
        
        if not self.data:
            raise ValueError("No data loaded.")

        # Validate bounds
        shape = self.metadata["size"][0]
        axis_order = self.metadata["axes"].lower()
        for name, index in zip("tczyx", [T, C, Z, Y, X]):
            if index is not None:
                max_len = shape[axis_order.index(name)]
                if max(index) >= max_len or min(index) < 0:
                    raise ValueError(f"Index for {name.upper()} out of range.")

        # Subset data
        self.data = [
            subset_dask_array(level, T=T, C=C, Z=Z, Y=Y, X=X)
            for level in self.data
        ]

        # Subset metadata
        self.metadata = subset_metadata(self.metadata, T=T, C=C, Z=Z, Y=Y, X=X)

        print("✅ Dataset subset complete.")
            

