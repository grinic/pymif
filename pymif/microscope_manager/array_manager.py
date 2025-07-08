from typing import Tuple, List, Dict, Any, Union
import dask.array as da
import numpy as np
import warnings
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
        chunks: Tuple[int, ...] = (1, 1, 8, 4096, 4096),
    ):
        """
        Initialize ArrayManager with array(s) and metadata.

        Parameters
        ----------
        array : Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]]
            A single array or a list of arrays (NumPy or Dask), shape (T,C,Z,Y,X)
        metadata : Dict[str, Any]
            NGFF-style metadata dictionary
        chunks : Tuple[int, ...]
            Desired chunk shape for Dask (if array is NumPy or unchunked)
        """
        super().__init__()
        self.data = array
        self.metadata = metadata
        self.chunks = chunks
        self.read()


    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Return the pyramid and metadata.

        Returns
        ----------
        Tuple[List[da.Array], Dict[str, Any]]
            A tuple containing a list of
            Dask arrays representing image data and a dictionary of metadata.
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

        # Validate and normalize scales
        user_scales = metadata.get("scales", [])
        if isinstance(user_scales, list) and len(user_scales) == len(self.data):
            scales = user_scales
        else:
            if user_scales:
                warnings.warn(
                    f"⚠️ Metadata 'scales' length ({len(user_scales)}) does not match pyramid levels ({len(self.data)}). Falling back to automatic scale generation."
                )
            base_scale = tuple(user_scales[0]) if user_scales else (1.0, 1.0, 1.0)
            scales = [tuple(s * (2**i) for s in base_scale) for i in range(len(self.data))]
            
        self.metadata["scales"] = scales
            
        self.metadata.setdefault("units", ("micrometer", "micrometer", "micrometer"))
        self.metadata.setdefault(
            "channel_names",
            [f"Channel {i}" for i in range(self.data[0].shape[1])]
        )
        self.metadata.setdefault("channel_colors", [0xFFFFFF] * self.data[0].shape[1])
        self.metadata.setdefault("time_increment", 1.0)
        self.metadata.setdefault("time_increment_unit", "s")
        self.metadata.setdefault("plane_files", None)

        return
