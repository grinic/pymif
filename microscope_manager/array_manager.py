from typing import Tuple, List, Dict, Any, Union
import dask.array as da
import numpy as np
from .microscope_manager import MicroscopeManager


class ArrayManager(MicroscopeManager):
    """
    Create a MicroscopeManager instance from in-memory NumPy or Dask array(s)
    with user-defined metadata. Supports single resolution or multiscale pyramid.
    """

    def __init__(
        self,
        array: Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]],
        metadata: Dict[str, Any],
        chunks: Tuple[int, ...] = (1, 1, 16, 256, 256),
    ):
        """
        Initialize ArrayManager with array(s) and metadata.

        Parameters:
        - array: A single array or a list of arrays (NumPy or Dask), shape (T,C,Z,Y,X)
        - metadata: NGFF-style metadata dictionary
        - chunks: Desired chunk shape for Dask (if array is NumPy or unchunked)
        """
        super().__init__()
        self.data = array
        self.metadata = metadata
        self.chunks = chunks
        self.read()


    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Return the pyramid and metadata.

        Returns:
        - List of dask arrays (one per resolution level)
        - Metadata dictionary
        """
        array = self.data
        metadata = self.metadata
        chunks = self.chunks

        # Wrap into list if input is a single array
        if not isinstance(array, list):
            array = [array]

        self.data = []
        for level in array:
            if isinstance(level, np.ndarray):
                level = da.from_array(level, chunks=chunks)
            elif isinstance(level, da.Array) and not level.chunks:
                level = level.rechunk(chunks)

            if level.ndim != 5:
                raise ValueError("Each level must have shape (T, C, Z, Y, X)")
            self.data.append(level)

        self.metadata = metadata

        # Fallbacks and normalization
        self.metadata.setdefault("axes", "tczyx")
        self.metadata.setdefault("dtype", str(self.data[0].dtype))
        self.metadata.setdefault("size", [level.shape for level in self.data])

        # Ensure metadata has proper 'scales' per level
        if "scales" not in self.metadata or not isinstance(self.metadata["scales"], list):
            # Assume base scale if not present
            base_scale = (1.0, 1.0, 1.0)
        else:
            base_scale = self.metadata["scales"][0]

        # Normalize to list of tuples (one per pyramid level)
        self.metadata["scales"] = [
            tuple(s * (2**i) for s in base_scale)
            for i in range(len(self.data))
        ]
        
        self.metadata.setdefault("units", ("micrometer", "micrometer", "micrometer"))
        self.metadata.setdefault(
            "channel_names",
            [f"Channel {i}" for i in range(self.data[0].shape[1])]
        )
        self.metadata.setdefault("channel_colors", [0xFFFFFF] * self.data[0].shape[1])
        self.metadata.setdefault("time_increment", 1.0)
        self.metadata.setdefault("time_increment_unit", "s")
        self.metadata.setdefault("plane_files", None)

        return self.data, self.metadata
