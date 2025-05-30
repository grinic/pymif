import os
import dask.array as da
from tifffile import imread
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dask import delayed
from .microscope_manager import MicroscopeManager

class OperaManager(MicroscopeManager):
    
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        """
        Initialize the ViventisManager with the given file path.

        Args:
            path (str): Path to the Viventis data file.
        """
        
        super().__init__()
        self.position_dir = Path(path)
        self.chunks = chunks
        self.read()

    def _parse_metadata(self) -> Dict[str, Any]:
        
        return None
        # return {
        #     "size": [(size_t, size_c, size_z, size_y, size_x)],
        #     "scales": scales,
        #     "units": units,
        #     "time_increment": time_increment,
        #     "time_increment_unit": time_increment_unit,
        #     "channel_names": channel_names,
        #     "channel_colors": channel_colors,
        #     "dtype": dtype,
        #     "plane_files": plane_files,
        #     "axes": "tczyx"
        # }

    def _build_dask_array(self) -> List[da.Array]:
  
        return None
        # return [stack]  # level 0 only

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read the Viventis data file and extract image arrays and metadata.

        Returns:
            Tuple[List[da.Array], Dict[str, Any]]: A tuple containing a list of
            Dask arrays representing image data and a dictionary of metadata.
        """
        
        self.metadata = self._parse_metadata()
        self.data = self._build_dask_array()
        return (self.data, self.metadata)
    
