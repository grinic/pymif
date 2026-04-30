from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import warnings

import dask.array as da
import numpy as np

from .microscope_manager import MicroscopeManager
from .utils.axes import normalize_axes, infer_axes_from_ndim, spatial_axes_in_order, normalize_data_type


class ArrayManager(MicroscopeManager):
    """
    Create a MicroscopeManager instance from in-memory NumPy or Dask array(s)
    with user-defined metadata. Supports single resolution or multiscale pyramid.

    The default remains legacy TCZYX for 5D arrays, but ``metadata['axes']`` may
    be any unique combination of ``t``, ``c``, ``z``, ``y`` and ``x`` whose length
    matches the input arrays.
    """

    def __init__(
        self,
        array: Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]],
        metadata: Dict[str, Any],
        chunks: Tuple[int, ...] = (1, 1, 8, 4096, 4096),
    ):
        """Initialize ArrayManager with a single array or pyramid.

        Parameters
        ----------
        array
            A NumPy/Dask array or list of arrays.  All levels must have the same
            dimensionality.
        metadata
            Metadata dictionary.  ``axes`` may be any subset of ``tczyx`` and
            ``data_type`` may be ``"intensity"`` or ``"label"``.
        chunks
            Dask chunk shape for NumPy inputs.  When omitted or incompatible with
            the dimensionality, automatic chunking is used.
        """
        super().__init__()
        self.data = array
        self.metadata = dict(metadata)
        self.chunks = chunks
        self.read()

    @staticmethod
    def _normalize_chunks(chunks, shape: tuple[int, ...]):
        if chunks is None:
            return "auto"
        if chunks == "auto":
            return "auto"
        try:
            chunk_tuple = tuple(int(c) for c in chunks)
        except TypeError:
            return "auto"
        if len(chunk_tuple) != len(shape):
            return "auto"
        return tuple(max(1, min(int(c), int(s))) for c, s in zip(chunk_tuple, shape))

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """Return the pyramid and metadata.

        Returns
        ----------
        Tuple[List[da.Array], Dict[str, Any]]
            A tuple containing a list of
            Dask arrays representing image data and a dictionary of metadata.
        """
        array = self.data
        metadata = dict(self.metadata)
        chunks = self.chunks

        if not isinstance(array, list):
            array = [array]

        if not array:
            raise ValueError("array cannot be empty.")

        axes = normalize_axes(metadata.get("axes") or infer_axes_from_ndim(array[0].ndim), ndim=array[0].ndim)
        self.data = []
        for level in array:
            if getattr(level, "ndim", None) != len(axes):
                raise ValueError(
                    f"Each level ndim must match metadata['axes']={''.join(axes)!r}."
                )
            if isinstance(level, np.ndarray):
                level = da.from_array(level, chunks=self._normalize_chunks(chunks, level.shape))
            elif isinstance(level, da.Array):
                if not level.chunks:
                    level = level.rechunk(self._normalize_chunks(chunks, level.shape))
            else:
                raise TypeError("array levels must be NumPy arrays or Dask arrays.")
            self.data.append(level)

        metadata["axes"] = "".join(axes)
        metadata.setdefault("dtype", str(self.data[0].dtype))
        metadata["data_type"] = normalize_data_type(metadata.get("data_type"))
        metadata.setdefault("size", [tuple(level.shape) for level in self.data])
        metadata.setdefault("chunksize", [tuple(level.chunksize) for level in self.data])

        spatial_axes = spatial_axes_in_order(axes)
        user_scales = metadata.get("scales", [])
        if isinstance(user_scales, list) and len(user_scales) == len(self.data):
            scales = [tuple(scale) for scale in user_scales]
        else:
            if user_scales:
                warnings.warn(
                    "Metadata 'scales' length does not match pyramid levels. "
                    "Falling back to automatic scale generation.",
                    stacklevel=2,
                )
            base_scale = tuple(user_scales[0]) if user_scales else tuple(1.0 for _ in spatial_axes)
            if len(base_scale) != len(spatial_axes):
                base_scale = tuple(1.0 for _ in spatial_axes)
            scales = [tuple(float(s) * (2**i) for s in base_scale) for i in range(len(self.data))]
        metadata["scales"] = scales

        metadata.setdefault("units", tuple("micrometer" for _ in spatial_axes))
        if "c" in axes:
            c_size = int(self.data[0].shape[axes.index("c")])
            metadata.setdefault("channel_names", [f"Channel {i}" for i in range(c_size)])
            metadata.setdefault("channel_colors", ["FFFFFF"] * c_size)
        else:
            metadata.setdefault("channel_names", [])
            metadata.setdefault("channel_colors", [])

        if "t" in axes:
            metadata.setdefault("time_increment", 1.0)
            metadata.setdefault("time_increment_unit", "s")
        else:
            metadata.setdefault("time_increment", None)
            metadata.setdefault("time_increment_unit", None)
        metadata.setdefault("plane_files", None)

        self.metadata = metadata
        return self.data, self.metadata
