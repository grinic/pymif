from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from .microscope_manager import MicroscopeManager
# from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziReader
# from bioio_czi.pylibczirw_reader.reader import Reader as PyLibCziReader
from bioio import BioImage
import numpy as np

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
        
        czi = BioImage(path, reconstruct_mosaic=True, use_aicspylibczi=False)
        self.scenes = czi.scenes
        if scene_name == "":
            self.scene_index = scene_index
            assert scene_index<len(czi.scenes), ValueError(f"Invalid scene index {scene_index}, only {len(czi.scenes)} scenes available: {czi.scenes}")
            self.scene_name = czi.scenes[scene_index]
        else:
            assert scene_name in czi.scenes, ValueError(f"Invalid scene {scene_name}: scene not found in available scenes: {czi.scenes}")
            self.scene_name = scene_name
            self.scene_index = czi.scenes.index(scene_name)
            
        print(f"Scenes: {czi.scenes}, loading {czi.scenes[self.scene_index]}. Rerun `read(scene_index)` to load another scene.")

        self.chunks = chunks
        self.read( scene_index = self.scene_index )
        
    def read(self,
             scene_index: int = 0):
        """
        Read the Zeiss dataset and populate self.data and self.metadata.

        Returns
        -------
        Tuple[List[da.Array], Dict[str, Any]]
            A tuple containing:
            - A list with one dask array representing the image data.
            - A metadata dictionary with pixel sizes, units, axes, etc.
        """
                
        czi = BioImage(self.path, reconstruct_mosaic=True, use_aicspylibczi=False)

        assert scene_index<len(czi.scenes), ValueError(f"Invalid scene index {scene_index}, only {len(czi.scenes)} scenes available: {czi.scenes}")
        self.scene_index = scene_index
        self.scene_name = czi.scenes[scene_index]
            
        czi.set_scene(self.scene_index)
        if self.chunks is None:
            self.chunks = czi.get_image_dask_data("TCZYX").chunksize
        self.data = [ czi.get_image_dask_data("TCZYX").rechunk(self.chunks) ]
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
        
        czi = BioImage(self.path, reconstruct_mosaic=True, use_aicspylibczi=False)
        czi.set_scene(self.scene_index)
        
        size = czi.get_image_dask_data("TCZYX").shape
        
        # The values are stored in units of meters always in .czi. Convert to microns.
        try:
            pxl_z = float(czi.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='Z']")[0].find("./Value").text)/1e-6
        except:
            pxl_z  =1.
        scales = tuple([
            pxl_z,
            float(czi.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='Y']")[0].find("./Value").text)/1e-6,
            float(czi.metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='X']")[0].find("./Value").text)/1e-6,
        ])
        
        units = ["micrometer"] * 3
        
        time_increment = np.clip( float(czi.metadata.findtext(".//TimeSeriesSetup/Interval/TimeSpan/Value") or 1.0), 1.0, None )
        time_unit = czi.metadata.findtext(".//TimeSeriesSetup/Interval/TimeSpan/DefaultUnitFormat") or "s"

        # Channels
        colors = []
        default_colors = [0xFF0000, 0x0000FF, 0x00FF00]
        for i, ch in enumerate( czi.metadata.findall(".//DisplaySetting/Channels/Channel") ):
            color = ch.findtext("Color") or default_colors[i%len(default_colors)]
            color = str(color)
            if (len(color) == 9) and (color[0] == "#"):
                # AARRGGBB → drop AA
                color = "#"+color[3:]
            colors.append(color)
        
        bit_depth = int(czi.metadata.findtext(".//BitsPerPixel") or 16)
        if bit_depth==8:
            dtype = "uint8"
        elif bit_depth==16:
            dtype = "uint16"
        else:
            dtype = "Unknown"
        
        return {
            "size": [size],
            "scales": [scales],
            "units": tuple(units),
            "time_increment": time_increment,
            "time_increment_unit": time_unit,
            "channel_names": [str(n) for n in czi.channel_names],
            "channel_colors": colors,  # Example, map from name if needed
            "dtype": dtype,
            "plane_files": Path(self.path).stem,
            "axes": "tczyx"
        }
