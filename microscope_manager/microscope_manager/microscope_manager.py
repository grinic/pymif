from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import dask.array as da


class MicroscopeManager(ABC):
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

    def write(self, path: str, compressor: Any = None) -> None:
        from .utils.write import write as _write
        return _write(path, self.data, self.metadata, compressor)

    def visualize(self) -> Any:
        from .utils.visualize import visualize as _visualize
        return _visualize(self.data, self.metadata)

    def build_pyramid(self, num_levels: int = 3, downscale_factor: int = 2) -> None:
        from .utils.pyramid import build_pyramid as _build_pyramid
        """
        Converts single-resolution data into a pyramidal multiscale structure
        and updates self.data and self.metadata in-place.
        """
        self.data, self.metadata = _build_pyramid(
            self.data[0], self.metadata, num_levels=num_levels, downscale_factor=downscale_factor
        )

    def close(self) -> None:
        """Close all open resources, such as file handles."""
        for f in getattr(self, "_open_files", []):
            try:
                f.close()
            except Exception as e:
                print(f"Warning: failed to close file: {e}")
        self._open_files = []

