from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import dask.array as da
from .microscope_manager import MicroscopeManager
import zarr
import napari
import os

class ZarrManager(MicroscopeManager):
    """
    A manager class for reading and handling OME-Zarr datasets.

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
                 mode: str = "r",
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
        self.mode = mode
        self.metadata = metadata
        
        # Access raw NGFF metadata

        # If path exists and we're in read mode, read it
        if os.path.exists(self.path):
            if mode in ("r", "a"):
                self.root = zarr.open(zarr.storage.LocalStore(self.path), mode=self.mode)
                self.read()
            else:
                raise FileNotFoundError(f"Zarr path {self.path} exists and mode='{mode}' is write-only. Please select mode='r' or 'a'.")
        else:
            if mode in ("w", "a"):
                self.root = zarr.open(zarr.storage.LocalStore(self.path), mode=self.mode)
                from .utils.create_empty_dataset import create_empty_dataset as _create_empty_dataset
                _create_empty_dataset(self.root,
                                      self.metadata)
            else:
                raise FileNotFoundError(f"Zarr path {self.path} does not exist and mode='{mode}' is read-only. Please select mode='w' or 'a'.")
        
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
        
        image_meta = self.root.attrs.asdict().get("ome")
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
        multiscale = group.attrs.get("ome").get("multiscales", [{}])[0]
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
            label_multiscales = label_grp.attrs.get("ome").get("multiscales", [])
            if not label_multiscales:
                continue

            label_datasets = label_multiscales[0].get("datasets", [])
            labels[label_name] = [da.from_zarr(label_grp[ds["path"]]) for ds in label_datasets]
        return labels
        
    def add_label(self, 
                    label_levels: List[da.Array],
                    label_name: str = "new_label",
                    compressor: Any = None, 
                    compressor_level: Any = 3, 
                    parallelize: Any = False) -> None:
        """
        Add a label layer to the Zarr dataset.

        Parameters
        ----------
        label_levels : List[dask.array.Array]
            A list of arrays representing a label image pyramid.
        label_name : str, optional
            Name of the label group (default is "new_label").
        compressor : Any, optional
            Compression method to use when writing labels.
        compressor_level : Any, optional
            Compression level to apply (default: 3).
        parallelize : bool, optional
            If True, label data is written in parallel.
        """
        
        from .utils.add_label import add_label as _add_label
        return _add_label(
                        root=self.root,
                        mode=self.mode, 
                        label_levels=label_levels, 
                        label_name=label_name,
                        metadata=self.metadata, 
                        compressor=compressor, 
                        compressor_level=compressor_level,
                        parallelize=parallelize
                      )
        
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
            image_meta = self.root.attrs.asdict().get("ome")
            if "multiscales" in image_meta:
                results.append(path)
            # print(results)
            # print(group.groups)
            for name, sub in group.groups():
                if "labels" not in name:
                    subpath = f"{path}/{name}" if path else name
                    results.extend(discover_multiscales(sub, path=subpath))
            # print(results)
            return results
        
        multiscale_paths = discover_multiscales(self.root)
        # print(multiscale_paths)
        
        for gpath in multiscale_paths:
            # open each group individually via napari-ome-zarr
            full_path = Path(self.path) / gpath
            # print("open", full_path)
            viewer.open(full_path, plugin="napari-ome-zarr") 
        
        return viewer
    
    def create_empty_group(
            self,
            group_name: str,
            metadata: Dict[str, Any],
            is_label: bool = False,
    ):
        from .utils.create_empty_group import create_empty_group as _create_empty_group
        return _create_empty_group(
            root = self.root,
            group_name = group_name,
            metadata = metadata,
            is_label = is_label,
        )

    def write_image_region(
        self,
        data,
        t: int | slice = slice(None),
        c: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: Optional[str] = None,
    ):
        """
        Public method to write or update a region in the dataset or sub-group.

        Parameters
        ----------
        data : Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]]
            Arrays representing an image region or its pyramid.
        t, c, z, y, x : int or slice
            Slices for each dimension.
        level : int
            The pyramid level to write to (if `data` is a single array).
        group_name : str, optional
            Name of the group inside the root.       
        """
        from .utils.write_image_region import write_image_region as _write_image_region
        return _write_image_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t, c=c, z=z, y=y, x=x,
            level=level,
            group_name=group,
        )

    def write_label_region(
        self,
        data,
        t: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: str = None,
    ):
        """
        Public method to write or update a region in the dataset or sub-group.
        """
        from .utils.write_label_region import write_label_region as _write_label_region
        return _write_label_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t, z=z, y=y, x=x,
            level=level,
            group_name=group,
        )
        