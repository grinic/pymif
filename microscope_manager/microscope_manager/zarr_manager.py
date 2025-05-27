from pathlib import Path
from typing import Tuple, List, Dict, Any
import dask.array as da
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


class ZarrManager():
    
    def __init__(self, 
                 path,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        super().__init__()
        self.store = parse_url(path)
        self.data, self.metadata = self.read()
        
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read an existing OME-Zarr dataset and extract data and metadata.
        """
        reader = Reader(self.store)
        nodes = list(reader())
        if not nodes:
            raise ValueError(f"No readable nodes found in {self.root}")
        
        node = nodes[0]
        data_levels = node.data
        image_meta = node.metadata

        axes = [axis.name for axis in image_meta["axes"]]
        scales = []
        for dataset in image_meta.datasets:
            scale = []
            for transform in dataset.coordinateTransformations:
                if transform["type"] == "scale":
                    scale = transform["scale"]
                    break
            scales.append(scale)

        channel_names = []
        channel_colors = []
        exposure_times = []
        laser_lines = []

        if image_meta.omero:
            for ch in image_meta.omero.channels:
                channel_names.append(ch.label)
                channel_colors.append(ch.color)
                exposure_times.append(getattr(ch, "exposureTime", None))
                laser_lines.append(getattr(ch, "wavelengthId", None))

        metadata = {
            "axes": axes,
            "scales": scales,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "exposure_times": exposure_times,
            "laser_lines": laser_lines,
            "name": image_meta.name or "OME-Zarr image",
        }

        return data_levels, metadata    
