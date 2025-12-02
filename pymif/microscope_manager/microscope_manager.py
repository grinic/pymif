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

        Returns
        ----------
        Tuple
            - List of Dask arrays for each resolution level.
            - Dictionary of extracted metadata.
        """
        pass

    def to_zarr(self, path: str, 
              compressor: Any = None, 
              compressor_level: Any = 3, 
              overwrite=True,
              parallelize: Any = False) -> None:
        from .utils.to_zarr import to_zarr as _to_zarr
        return _to_zarr(path, 
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

        Parameters
        ----------
            new_order : List[int]
                A permutation of the channel indices.
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
                
            if key == "channel_colors":
                value = [parse_color(v) for v in value]

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
        import numpy as np
        
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

        num_levels = len(self.data)
        downscale_factor = 2
        if num_levels>1:
            downscale_factor = int(np.round(self.metadata["size"][0][-1]/self.metadata["size"][1][-1]))

        # Subset data
        self.data = [subset_dask_array(self.data[0], T=T, C=C, Z=Z, Y=Y, X=X)]

        # Subset metadata
        self.metadata = subset_metadata(self.metadata, T=T, C=C, Z=Z, Y=Y, X=X)
        
        # rebuild pyramid
        self.build_pyramid(
            num_levels = num_levels,
            downscale_factor = downscale_factor,
        )

        print("Dataset subset complete.")
            

