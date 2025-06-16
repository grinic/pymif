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
                 chunks: Tuple[int, ...] = None,
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
            assert scene_index<len(aics.scenes), ValueError(f"Invalid scene index {scene_index}, only {len(aics.scenes)} scenes available: {aics.scenes}")
            self.scene_name = aics.scenes[scene_index]
        else:
            assert scene_name in aics.scenes, ValueError(f"Invalid scene {scene_name}: scene not found in available scenes: {aics.scenes}")
            self.scene_name = scene_name
            self.scene_index = aics.scenes.index(scene_name)

        self.aics = aics

        self.chunks = chunks
        self.read()
        
    def read(self):
        if self.chunks is None:
            self.chunks = self.aics.get_image_dask_data("TCZYX").chunksize
        self.data = self.aics.get_image_dask_data("TCZYX").rechunk(self.chunks)
        self.metadata = self._parse_metadata()
        return
        
    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse metadata from the .czi dataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
        """
        
        size = tuple([
            self.aics.standard_metadata.image_size_t,
            self.aics.standard_metadata.image_size_c,
            self.aics.standard_metadata.image_size_z,
            self.aics.standard_metadata.image_size_y,
            self.aics.standard_metadata.image_size_x,
        ])
        
        scales = tuple([
            self.aics.standard_metadata.pixel_size_z,
            self.aics.standard_metadata.pixel_size_y,
            self.aics.standard_metadata.pixel_size_x,
        ])
        
        units = tuple([
            self.aics.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='Z']")[0].find("./DefaultUnitFormat").text,
            self.aics.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='Y']")[0].find("./DefaultUnitFormat").text,
            self.aics.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='X']")[0].find("./DefaultUnitFormat").text,
        ])
        
        time_increment = float(self.aics.metadata.findtext(".//TimeSeriesSetup/Interval/TimeSpan/Value") or 1.0)
        time_unit = self.aics.metadata.findtext(".//TimeSeriesSetup/Interval/TimeSpan/DefaultUnitFormat") or "s"

        # Channels
        # channels = []
        colors = []
        default_colors = [0xFF0000, 0x0000FF]
        for i, ch in enumerate( self.aics.metadata.findall(".//Channels/Channel") ):
            # name = ch.findtext(".//AdditionalDyeInformation/ShortName") or ch.findtext(".//FluorescenceDye/ShortName") or "Unnamed"
            color = ch.findtext("Color") or default_colors[i]
            # channels.append(name)
            colors.append(color)
        
        bit_depth = int(self.aics.metadata.findtext(".//BitsPerPixel") or 16)
        
        return {
            "size": [size],
            "scales": scales,
            "units": tuple(units),
            "time_increment": time_increment,
            "time_increment_unit": time_unit,
            "channel_names": self.aics.channel_names,
            "channel_colors": colors,  # Example, map from name if needed
            "dtype": bit_depth,
            "plane_files": None,
            "axes": "tczyx"
        }
