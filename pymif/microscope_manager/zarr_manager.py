from pathlib import Path
from typing import Tuple, List, Dict, Any
import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from .microscope_manager import MicroscopeManager
import zarr
from zarr.storage import KVStore

class ZarrManager(MicroscopeManager):
    
    def __init__(self, 
                 path,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        super().__init__()
        self.store = parse_url(path)
        self.chunks = chunks
        self.read()
        
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read an existing OME-Zarr dataset and extract data and metadata.
        
        Returns:
            Tuple[List[da.Array], Dict[str, Any]]: A tuple containing a list of
            Dask arrays representing image data and a dictionary of metadata.
        """
        reader = Reader(self.store)
        nodes = list(reader())
        if not nodes:
            raise ValueError(f"No readable nodes found in {self.root}")
        
        node = nodes[0]
        
        # Access raw NGFF metadata
        group = zarr.open(KVStore(self.store.store), mode="r")
        image_meta = group.attrs.asdict()
        multiscales = image_meta.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])
        omero = image_meta.get("omero", {})
        data_levels = node.data
        dtype = data_levels[0].dtype
        
        # Metadata
        axes = "".join([a["name"][0] for a in multiscales.get("axes", [])])
        sizes = [ tuple(d.shape) for d in data_levels ]
        scales = [tuple(d.get("coordinateTransformations", [{}])[0].get("scale", None)[2:])
                  for d in datasets]
        units = tuple([a.get("unit", None) for a in multiscales.get("axes", [])][2:])

        # Channels
        channels = omero.get("channels", [])
        channel_names = [ch.get("label", f"Channel {i}") for i, ch in enumerate(channels)]
        channel_colors = [ch.get("color", "FFFFFF") for ch in channels]
        channel_colors = [int(c, 16) if isinstance(c, str) else c for c in channel_colors]

        # Time increment
        time_increment = datasets[0].get("coordinateTransformations", [{}])[0].get("scale", None)[0]
        time_increment_unit = [a.get("unit", None) for a in multiscales.get("axes", [])][0]

        self.metadata = {
            "size": sizes,
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_increment_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": str(dtype),
            "plane_files": None,
            "axes": axes
        }

        # Dask pyramids
        pyramid = data_levels#[da.from_zarr(Path(self.path) / ds["path"]) for ds in datasets]
        pyramid = [p.rechunk(self.chunks) for p in pyramid]

        self.data = pyramid
        return self.data, self.metadata