from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from .microscope_manager import MicroscopeManager
import zarr
from zarr.storage import KVStore
import napari

class ZarrManager(MicroscopeManager):
    
    def __init__(self, 
                 path,
                 chunks: Tuple[int, ...] = None):
        super().__init__()
        self.path = path
        self.chunks = chunks
        self.read()
        
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read an existing OME-Zarr dataset and extract data and metadata.
        
        Returns:
            Tuple[List[da.Array], Dict[str, Any]]: A tuple containing a list of
            Dask arrays representing image data and a dictionary of metadata.
        """
        # reader = Reader(self.path)
        # nodes = list(reader())
        # if not nodes:
        #     raise ValueError(f"No readable nodes found in {self.root}")
        
        # node = nodes[0]
        
        # Access raw NGFF metadata
        group = zarr.open(self.path, mode="r")
        image_meta = group.attrs.asdict()
        multiscales = image_meta.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])
        omero = image_meta.get("omero", {})
        
        # Load pyramid levels properly
        data_levels = []
        for i in range(len(datasets)):
            zarr_array = group[str(i)]
            if self.chunks is None:
                arr = da.from_zarr(zarr_array)  # uses native chunking
            else:
                arr = da.from_zarr(zarr_array, chunks=self.chunks) # use chunks (same for all levels)
            data_levels.append(arr)      
        self.chunks = data_levels[0].chunksize
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

        self.data = data_levels
        
        ### optional, read labels
        self.labels = self._load_labels()

    def _load_labels(self) -> Dict[str, List[da.Array]]:
        """
        Load labels stored under `/labels/<label_name>/` groups.

        Returns:
            Dict[label_name, List[dask.Array]]: each label name maps to a list of dask arrays (one per pyramid level)
        """
        labels = {}
        root = zarr.open_group(str(self.path), mode='r')
        if "labels" not in root:
            return labels

        labels_grp = root["labels"]
        for label_name, label_grp in labels_grp.groups():
            # Expect label_grp to have a multiscale structure similar to images
            label_multiscales = label_grp.attrs.get("multiscales", [])
            if not label_multiscales:
                continue

            label_datasets = label_multiscales[0].get("datasets", [])
            label_pyramid = []
            for ds in label_datasets:
                ds_path = ds["path"]
                zarr_arr = label_grp[ds_path]
                label_pyramid.append(da.from_zarr(zarr_arr))
            labels[label_name] = label_pyramid
        return labels
        
    def add_label(self, 
                    label_levels: List[da.Array],
                    label_name: str = "new_label",
                    compressor: Any = None, 
                    compressor_level: Any = 3, 
                    parallelize: Any = False) -> None:
        from .utils.add_label import add_label as _add_label
        return _add_label(self.path, 
                      label_levels, 
                      label_name,
                      self.metadata, 
                      compressor=compressor, 
                      compressor_level=compressor_level,
                      parallelize=parallelize
                      )
        
    def visualize_zarr(self,
            viewer: Optional[napari.Viewer] = None,
        ) -> napari.Viewer:
        
        if viewer is None:
            viewer = napari.Viewer()
        
        viewer.open(self.path, plugin="napari-ome-zarr")
        
        return viewer
        
        
