from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import dask.array as da
from .microscope_manager import MicroscopeManager
import zarr
import napari
import os

class ZarrV04Manager(MicroscopeManager):
    """
    A manager class for reading and handling OME-Zarr V0.4 datasets.

    This class reads NGFF-compliant multiscale image data along with
    metadata and optional label layers from a Zarr directory. It supports
    lazy loading with Dask, label layer detection, and integration with
    Napari for visualization.

    Parameters
    ----------
    path : str or Path
        Path to the root directory of the OME-Zarr dataset.
    chunks : Tuple[int, ...], optional
        Chunk size to apply when reading arrays. If None, native chunking
        in the Zarr arrays is used.
    """
        
    def __init__(self, 
                 path,
                 chunks: Tuple[int, ...] = None,
                 metadata: dict[str, Any] = None):
        """
        Initialize the ZarrManager.

        Parameters
        ----------
        path : str
            Path to the folder containing the Zarr dataset.
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Default is `None`.
        mode : {"r", "r+", "a", "w", "w-"}, default="r"
            Zarr access mode. Use "r+" to enable in-place modification.        
        """
        
        super().__init__()
        self.path = path
        self.chunks = chunks
        self.mode = "r"
        self.metadata = metadata
        
        # Access raw NGFF metadata

        # If path exists and we're in read mode, read it
        self.root = zarr.open(zarr.storage.LocalStore(self.path), mode=self.mode)
        self.read()
        
    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read an OME-Zarr dataset and extract both image data and metadata.

        This method lazily loads the multiscale image pyramid and metadata
        from the root of the Zarr dataset. If `/labels/` groups exist, they
        are also read into the `self.labels` attribute.

        Returns
        -------
        data_levels : List[dask.array.Array]
            List of Dask arrays, one for each resolution level.
        metadata : Dict[str, Any]
            Dictionary containing image metadata such as axes, scales, sizes,
            channels, time increments, and units.

        Raises
        ------
        ValueError
            If the dataset structure is invalid or lacks required metadata.
        """
        
        image_meta = self.root.attrs.asdict()
        multiscales = image_meta.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])
        omero = image_meta.get("omero", {})
        
        # Load pyramid levels properly
        data_levels = []
        for i in range(len(datasets)):
            zarr_array = self.root[str(i)]
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
        chunksize = [d.chunksize for d in data_levels]
        units = tuple([a.get("unit", None) for a in multiscales.get("axes", [])][2:])

        # Channels
        channels = omero.get("channels", [])
        channel_names = [ch.get("label", f"Channel {i}") for i, ch in enumerate(channels)]
        channel_colors = [ch.get("color", "FFFFFF") for ch in channels]
        # channel_colors = [int(c, 16) if isinstance(c, str) else c for c in channel_colors]

        # Time increment
        time_increment = datasets[0].get("coordinateTransformations", [{}])[0].get("scale", None)[0]
        time_increment_unit = [a.get("unit", None) for a in multiscales.get("axes", [])][0]

        self.metadata = {
            "size": sizes,
            "chunksize": chunksize,
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_increment_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": str(dtype),
            "plane_files": None,
            "axes": axes,
        }

        self.data = data_levels
        
        ### optional, read other groups
        self.groups = {}
        self.labels = {}
        for name in self.root.group_keys():
            if name == "labels":
                self.labels = self._load_labels()
            else:
                self.groups[name] = self._load_group(name)
                
        ### Output
        print(self.root.tree())
        for i in self.metadata:
            print(f"{i.upper()}: {self.metadata[i]}")

    def _load_group(self, name):
        group = self.root[name]
        multiscale = group.attrs.get("multiscales", [{}])[0]
        datasets = multiscale.get("datasets", [])
        arrays = [da.from_zarr(group[ds["path"]]) for ds in datasets]
        return arrays

    def _load_labels(self) -> Dict[str, List[da.Array]]:
        """
        Load label layers from the `/labels/` group in the Zarr dataset.

        Each label is stored as a multiscale pyramid similar to image data.

        Returns
        -------
        labels : Dict[str, List[dask.array.Array]]
            Dictionary mapping each label name to its list of pyramid levels.
        """
        
        labels = {}
        # root = zarr.open_group(str(self.path), mode='r')
        if "labels" not in self.root:
            return labels

        labels_grp = self.root["labels"]
        for label_name, label_grp in labels_grp.groups():
            # Expect label_grp to have a multiscale structure similar to images
            label_multiscales = label_grp.attrs.get("multiscales", [])
            if not label_multiscales:
                continue

            label_datasets = label_multiscales[0].get("datasets", [])
            labels[label_name] = [da.from_zarr(label_grp[ds["path"]]) for ds in label_datasets]
        return labels
        
    def visualize_zarr(self,
            viewer: Optional[napari.Viewer] = None,
        ) -> napari.Viewer:
        """
        Visualize the OME-Zarr dataset using Napari's `napari-ome-zarr` plugin.

        Parameters
        ----------
        viewer : napari.Viewer, optional
            An existing viewer instance. If None, a new one will be created.

        Returns
        -------
        viewer : napari.Viewer
            A Napari viewer instance with the image loaded.
        """
                
        if viewer is None:
            viewer = napari.Viewer()
        
        # recursively find all groups with a "multiscales" attribute
        def discover_multiscales(group: zarr.Group, path=""):
            results = []
            if "multiscales" in group.attrs:
                results.append(path)
            print(results)
            print(group.groups)
            for name, sub in group.groups():
                if "labels" not in name:
                    subpath = f"{path}/{name}" if path else name
                    results.extend(discover_multiscales(sub, path=subpath))
            print(results)
            return results
        
        multiscale_paths = discover_multiscales(self.root)
        print(multiscale_paths)
        
        for gpath in multiscale_paths:
            # open each group individually via napari-ome-zarr
            full_path = Path(self.path) / gpath
            print("open", full_path)
            viewer.open(full_path, plugin="napari-ome-zarr") 
        
        return viewer
    