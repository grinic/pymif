import os
import dask.array as da
from tifffile import imread
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dask import delayed
from .microscope_manager import MicroscopeManager

class ViventisManager(MicroscopeManager):
    """
    Reader for Viventis microscope datasets with OME-TIFF and companion .ome XML files.
    
    This class lazily loads data into a dask array and parses associated OME-XML metadata.
    """
        
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        """
        Initialize the ViventisManager.

        Parameters
        ----------
        path : str
            Path to the folder containing the Viventis dataset (including `.ome` and `.tif` files).
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Default is `(1, 1, 16, 256, 256)`.
        """
        
        super().__init__()
        self.path = Path(path)
        self.chunks = chunks
        self.read()

    def _parse_companion_file(self) -> Dict[str, Any]:
        """
        Parse the companion `.ome` XML metadata file.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted metadata such as size, scales, units, channel info,
            time increment, and TIFF file mapping.
        """
        
        companion = list(self.path.glob("*.ome"))[0]
        tree = ET.parse(companion)
        root = tree.getroot()

        pixels = root.find(".//{*}Pixels")
        size_t = int(pixels.attrib["SizeT"])
        size_z = int(pixels.attrib["SizeZ"])
        size_y = int(pixels.attrib["SizeY"])
        size_x = int(pixels.attrib["SizeX"])
        size_c = int(pixels.attrib["SizeC"])

        scales = [(float(pixels.attrib["PhysicalSizeZ"]),
                   float(pixels.attrib["PhysicalSizeY"]),
                   float(pixels.attrib["PhysicalSizeX"]))]

        units = (pixels.attrib["PhysicalSizeZUnit"],
                 pixels.attrib["PhysicalSizeYUnit"],
                 pixels.attrib["PhysicalSizeXUnit"])

        time_increment = float(pixels.attrib.get("TimeIncrement", 1))
        time_increment_unit = pixels.attrib.get("TimeIncrementUnit", "s")

        channels = root.findall(".//{*}Channel")
        channel_names = [c.attrib.get("Name", f"Channel {i}") for i, c in enumerate(channels)]
        channel_colors = [int(c.attrib.get("Color", 0)) for c in channels]

        dtype = str(pixels.attrib["Type"])

        # Plane map: (t, c) -> filename
        tiffdata = root.findall(".//{*}TiffData")
        plane_files = {}
        for entry in tiffdata:
            t = int(entry.attrib["FirstT"])
            c = int(entry.attrib["FirstC"])
            z_count = int(entry.attrib.get("PlaneCount", 1))
            filename = entry.find(".//{*}UUID").attrib["FileName"]
            plane_files[(t, c)] = filename

        return {
            "size": [(size_t, size_c, size_z, size_y, size_x)],
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_increment_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": dtype,
            "plane_files": plane_files,
            "axes": "tczyx"
        }

    def _build_dask_array(self) -> List[da.Array]:
        """
        Lazily construct a dask array for the image data using tifffile and delayed loading.

        Returns
        -------
        List[da.Array]
            A list containing a single Dask array representing the full dataset (level 0).
        """
        
        t, c, z, y, x = self.metadata["size"][0]

        lazy_imread = delayed(imread)  # lazy reader
        filenames = self.metadata["plane_files"]
        lazy_arrays = [ [ lazy_imread(f"{self.path}/{filenames[(ti,ci)]}") for ci in range(c) ] for ti in range(t) ] 
        dask_arrays = [
            [
                da.from_delayed(l2, shape=(z,y,x), dtype=self.metadata["dtype"])
                for l2 in l1
            ]
            for l1 in lazy_arrays
        ]
        # Stack into one large dask.array
        stack = da.stack(dask_arrays, axis=0)
        stack = stack.rechunk((self.chunks))
  
        return [stack]  # level 0 only

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read the Viventis dataset and populate self.data and self.metadata.

        Returns
        -------
        Tuple[List[da.Array], Dict[str, Any]]
            A tuple containing:
            - A list with one dask array representing the image data.
            - A metadata dictionary with pixel sizes, units, axes, etc.
        """
        
        self.metadata = self._parse_companion_file()
        self.data = self._build_dask_array()
        return (self.data, self.metadata)
    
