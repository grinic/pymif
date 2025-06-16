from typing import Tuple, List, Dict, Any, Optional
import dask.array as da
from .microscope_manager import MicroscopeManager
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziReader
import zarr
import napari

class ZeissManager(MicroscopeManager):
    """
    A manager class for reading and handling .czi datasets.

    This class lazily loads data into a dask array and parses associated .czi metadata.

    """
        
    def __init__(self, 
                 path,
                 scene_index: int = 0,
                 scene_name: Optional[str] = "",
                 chunks: Tuple[int, ...] = None
                 ):
        """
        Initialize the ZarrManager.

        Parameters
        ----------
        path : str
            Path to the folder containing the Zarr dataset.
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Default is `None`.
        """
        
        super().__init__()
        self.path = path
        
        aics = AicsPyLibCziReader(path)
        if scene_name == "":
            self.scene_index = scene_index
            assert scene_index>len(aics.scenes), ValueError(f"Invalid scene index {scene_index}, only {len(aics.scenes)} scenes available: {aics.scenes}")
            self.scene_name = aics.scenes[scene_index]
        else:
            assert scene_name in aics.scenes, ValueError(f"Invalid scene {scene_name}: scene not found in available scenes: {aics.scenes}")
            self.scene_name = scene_name
            self.scene_index = aics.scenes.index(scene_name)

        self.chunks = chunks
        self.read()
        
    # def _parse_metadata(self) -> Dict[str, Any]:
    #     """
    #     Parse metadata from the .czi dataset.

    #     Returns
    #     -------
    #     Dict[str, Any]
    #         A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
    #     """
        
    #     return {
    #         "size": [],
    #         "scales": scales,
    #         "units": tuple(units),
    #         "time_increment": 1.0,
    #         "time_increment_unit": "s",
    #         "channel_names": channel_names,
    #         "channel_colors": [0xFF0000, 0x0000FF],  # Example, map from name if needed
    #         "dtype": "uint16",
    #         "plane_files": None,
    #         "axes": "tczyx"
    #     }
        
    # def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
    #     """
    #     Read an OME-Zarr dataset and extract both image data and metadata.

    #     This method lazily loads the multiscale image pyramid and metadata
    #     from the root of the Zarr dataset. If `/labels/` groups exist, they
    #     are also read into the `self.labels` attribute.

    #     Returns
    #     -------
    #     data_levels : List[dask.array.Array]
    #         List of Dask arrays, one for each resolution level.
    #     metadata : Dict[str, Any]
    #         Dictionary containing image metadata such as axes, scales, sizes,
    #         channels, time increments, and units.

    #     Raises
    #     ------
    #     ValueError
    #         If the dataset structure is invalid or lacks required metadata.
    #     """
        
    #     # Access raw NGFF metadata
    #     group = zarr.open(self.path, mode="r")
    #     image_meta = group.attrs.asdict()
    #     multiscales = image_meta.get("multiscales", [{}])[0]
    #     datasets = multiscales.get("datasets", [])
    #     omero = image_meta.get("omero", {})
        
    #     # Load pyramid levels properly
    #     data_levels = []
    #     for i in range(len(datasets)):
    #         zarr_array = group[str(i)]
    #         if self.chunks is None:
    #             arr = da.from_zarr(zarr_array)  # uses native chunking
    #         else:
    #             arr = da.from_zarr(zarr_array, chunks=self.chunks) # use chunks (same for all levels)
    #         data_levels.append(arr)      
    #     self.chunks = data_levels[0].chunksize
    #     dtype = data_levels[0].dtype
        
    #     # Metadata
    #     axes = "".join([a["name"][0] for a in multiscales.get("axes", [])])
    #     sizes = [ tuple(d.shape) for d in data_levels ]
    #     scales = [tuple(d.get("coordinateTransformations", [{}])[0].get("scale", None)[2:])
    #               for d in datasets]
    #     chunksize = [d.chunksize for d in data_levels]
    #     units = tuple([a.get("unit", None) for a in multiscales.get("axes", [])][2:])

    #     # Channels
    #     channels = omero.get("channels", [])
    #     channel_names = [ch.get("label", f"Channel {i}") for i, ch in enumerate(channels)]
    #     channel_colors = [ch.get("color", "FFFFFF") for ch in channels]
    #     channel_colors = [int(c, 16) if isinstance(c, str) else c for c in channel_colors]

    #     # Time increment
    #     time_increment = datasets[0].get("coordinateTransformations", [{}])[0].get("scale", None)[0]
    #     time_increment_unit = [a.get("unit", None) for a in multiscales.get("axes", [])][0]

    #     self.metadata = {
    #         "size": sizes,
    #         "chunksize": chunksize,
    #         "scales": scales,
    #         "units": units,
    #         "time_increment": time_increment,
    #         "time_increment_unit": time_increment_unit,
    #         "channel_names": channel_names,
    #         "channel_colors": channel_colors,
    #         "dtype": str(dtype),
    #         "plane_files": None,
    #         "axes": axes
    #     }

    #     self.data = data_levels
        
    #     ### optional, read labels
    #     self.labels = self._load_labels()
        
    #     def _default_label_color(name: str) -> str:
    #         name = name.lower()
    #         if "nuc" in name:
    #             return "magenta"
    #         elif "mem" in name:
    #             return "cyan"
    #         elif "mask" in name:
    #             return "yellow"
    #         return "white"
        
    #     if self.labels:
    #         self.metadata["labels_metadata"] = {}
    #         for label_name, label_levels in self.labels.items():
    #             # Try to read scale info from label multiscale metadata
    #             label_grp = zarr.open_group(str(self.path), mode='r')["labels"][label_name]
    #             label_multiscales = label_grp.attrs.get("multiscales", [])
    #             if label_multiscales:
    #                 label_datasets = label_multiscales[0].get("datasets", [])
    #                 if label_datasets and "coordinateTransformations" in label_datasets[0]:
    #                     label_scale = label_datasets[0]["coordinateTransformations"][0].get("scale", [1,1,1])[1:]
    #                 else:
    #                     label_scale = [1, 1, 1]
    #             else:
    #                 label_scale = [1, 1, 1]

    #             self.metadata["labels_metadata"][label_name] = {
    #                 "data": label_levels,
    #                 "scale": label_scale,
    #                 "color": _default_label_color(label_name),
    #                 "opacity": 0.5
    #             }

    # def _load_labels(self) -> Dict[str, List[da.Array]]:
    #     """
    #     Load label layers from the `/labels/` group in the Zarr dataset.

    #     Each label is stored as a multiscale pyramid similar to image data.

    #     Returns
    #     -------
    #     labels : Dict[str, List[dask.array.Array]]
    #         Dictionary mapping each label name to its list of pyramid levels.
    #     """
        
    #     labels = {}
    #     root = zarr.open_group(str(self.path), mode='r')
    #     if "labels" not in root:
    #         return labels

    #     labels_grp = root["labels"]
    #     for label_name, label_grp in labels_grp.groups():
    #         # Expect label_grp to have a multiscale structure similar to images
    #         label_multiscales = label_grp.attrs.get("multiscales", [])
    #         if not label_multiscales:
    #             continue

    #         label_datasets = label_multiscales[0].get("datasets", [])
    #         label_pyramid = []
    #         for ds in label_datasets:
    #             ds_path = ds["path"]
    #             zarr_arr = label_grp[ds_path]
    #             label_pyramid.append(da.from_zarr(zarr_arr))
    #         labels[label_name] = label_pyramid
    #     return labels
        
    # def add_label(self, 
    #                 label_levels: List[da.Array],
    #                 label_name: str = "new_label",
    #                 compressor: Any = None, 
    #                 compressor_level: Any = 3, 
    #                 parallelize: Any = False) -> None:
    #     """
    #     Add a label layer to the Zarr dataset.

    #     Parameters
    #     ----------
    #     label_levels : List[dask.array.Array]
    #         A list of arrays representing a label image pyramid.
    #     label_name : str, optional
    #         Name of the label group (default is "new_label").
    #     compressor : Any, optional
    #         Compression method to use when writing labels.
    #     compressor_level : Any, optional
    #         Compression level to apply (default: 3).
    #     parallelize : bool, optional
    #         If True, label data is written in parallel.
    #     """
        
    #     from .utils.add_label import add_label as _add_label
    #     return _add_label(self.path, 
    #                   label_levels, 
    #                   label_name,
    #                   self.metadata, 
    #                   compressor=compressor, 
    #                   compressor_level=compressor_level,
    #                   parallelize=parallelize
    #                   )
        
    # def visualize_zarr(self,
    #         viewer: Optional[napari.Viewer] = None,
    #     ) -> napari.Viewer:
    #     """
    #     Visualize the OME-Zarr dataset using Napari's `napari-ome-zarr` plugin.

    #     Parameters
    #     ----------
    #     viewer : napari.Viewer, optional
    #         An existing viewer instance. If None, a new one will be created.

    #     Returns
    #     -------
    #     viewer : napari.Viewer
    #         A Napari viewer instance with the image loaded.
    #     """
                
    #     if viewer is None:
    #         viewer = napari.Viewer()
        
    #     viewer.open(self.path, plugin="napari-ome-zarr")
        
    #     return viewer
        
        
