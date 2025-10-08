from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import dask.array as da
from .microscope_manager import MicroscopeManager
import zarr
import napari

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
                 mode: str = "r"):
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
        
        # Access raw NGFF metadata
        root = zarr.open(self.path, mode=self.mode)
        image_meta = root.attrs.asdict()
        multiscales = image_meta.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])
        omero = image_meta.get("omero", {})
        
        # Load pyramid levels properly
        data_levels = []
        for i in range(len(datasets)):
            zarr_array = root[str(i)]
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
        channel_colors = [int(c, 16) if isinstance(c, str) else c for c in channel_colors]

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
            "axes": axes
        }

        self.data = data_levels
        self.root = root  # keep handle for writing later
        
        ### optional, read labels
        self.labels = self._load_labels()
        
        def _default_label_color(name: str) -> str:
            name = name.lower()
            if "nuc" in name:
                return "magenta"
            elif "mem" in name:
                return "cyan"
            elif "mask" in name:
                return "yellow"
            return "white"
        
        if self.labels:
            self.metadata["labels_metadata"] = {}
            for label_name, label_levels in self.labels.items():
                # Try to read scale info from label multiscale metadata
                label_grp = zarr.open_group(str(self.path), mode='r')["labels"][label_name]
                label_multiscales = label_grp.attrs.get("multiscales", [])
                if label_multiscales:
                    label_datasets = label_multiscales[0].get("datasets", [])
                    if label_datasets and "coordinateTransformations" in label_datasets[0]:
                        label_scale = label_datasets[0]["coordinateTransformations"][0].get("scale", [1,1,1])[1:]
                    else:
                        label_scale = [1, 1, 1]
                else:
                    label_scale = [1, 1, 1]

                self.metadata["labels_metadata"][label_name] = {
                    "data": label_levels,
                    "scale": label_scale,
                    "color": _default_label_color(label_name),
                    "opacity": 0.5
                }

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
        
        viewer.open(self.path, plugin="napari-ome-zarr")
        
        return viewer
    
    def write_region(
        self,
        data: Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]],
        t: int | slice = slice(None),
        c: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: str = None,
    ):
        """
        Write or update a region of the dataset in-place.

        Parameters
        ----------
        data : np.ndarray or dask.array.Array or list thereof
            Array(s) to write. If a list, each entry corresponds to one
            resolution level.
        t, c, z, y, x : int or slice
            Indices or slices for each dimension.
        level : int
            The pyramid level to write to (if `data` is a single array).
        group : str, optional
            Zarr group to write into. If None, defaults to main image group.
            For labels, use e.g. "labels/my_label". 
        """
        if self.mode not in ("r+", "a", "w"):
            raise PermissionError(
                f"Dataset opened in read-only mode ('{self.mode}'). "
                "Reopen with mode='r+' to allow modifications."
            )

        if isinstance(data, (np.ndarray, da.Array)):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            raise TypeError("`data` must be a NumPy array, Dask array, or list of such.")
        
        target_group = self.root if group is None else self.root[group]
        multiscales = target_group.attrs["multiscales"][0]

        base_scale = np.array(self.metadata.get("scales", [[1, 1, 1]])[0])

        for i, subdata in enumerate(data_list):
            target_level = i
            if target_level >= len(multiscales["datasets"]):
                break
            arr_path = multiscales["datasets"][target_level]["path"]
            zarr_array = target_group[arr_path]

            if isinstance(subdata, da.Array):
                subdata = subdata.compute()

            # Scale the spatial indices according to the pyramid level
            scale = np.array(self.metadata["scales"][target_level]) / base_scale
            z_scale, y_scale, x_scale = scale[-3:]  # last three dims are spatial

            def _scale_index(s, factor):
                if isinstance(s, slice):
                    start = None if s.start is None else int(np.floor(s.start / factor))
                    stop = None if s.stop is None else int(np.ceil(s.stop / factor))
                    return slice(start, stop)
                elif isinstance(s, int):
                    return int(np.floor(s / factor))
                return s  
                      
            scaled_index = t, c, \
                _scale_index(z, z_scale), \
                _scale_index(y, y_scale), \
                _scale_index(x, x_scale)
            
            zarr_array[scaled_index] = subdata

