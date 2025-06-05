import os
import dask.array as da
import zarr
from tifffile import imread
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
import tifffile
from .microscope_manager import MicroscopeManager

class OperaManager(MicroscopeManager):
    
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        """
        Initialize the OperaManager with the given file path.

        Args:
            path (str): Path to the Opera data file.
        """
        
        super().__init__()
        self.path = Path(path)
        self.chunks = chunks
        self.read()

    def _parse_metadata(self) -> dict:
        
        with tifffile.TiffFile(self.path) as tif:
            xml_string = tif.ome_metadata
            
        # Parse OME-XML
        root = ET.fromstring(xml_string)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        pixels = root.find('.//ome:Pixels', ns)

        # Image size
        size_t = int(pixels.attrib.get("SizeT", 1))
        size_c = int(pixels.attrib.get("SizeC", 1))
        size_z = int(pixels.attrib.get("SizeZ", 1))
        size_y = int(pixels.attrib["SizeY"])
        size_x = int(pixels.attrib["SizeX"])

        # Physical pixel sizes
        px_x = float(pixels.attrib.get("PhysicalSizeX", 1))
        px_y = float(pixels.attrib.get("PhysicalSizeY", 1))
        px_z = float(pixels.attrib.get("PhysicalSizeZ", 1))
        unit = pixels.attrib.get("PhysicalSizeXUnit", "µm")

        # Time increment (not present in your example, but structured this way if available)
        time_increment = float(pixels.attrib.get("TimeIncrement", 1))
        time_unit = pixels.attrib.get("TimeIncrementUnit", "s")

        # Channel names and colors
        channel_names = []
        channel_colors = []
        for channel in pixels.findall("ome:Channel", ns):
            channel_names.append(channel.attrib.get("Name", ""))
            color = channel.attrib.get("Color")
            if color:
                channel_colors.append(int(color))
            else:
                channel_colors.append(0xFFFFFF)  # default white

        # Build scale list for NGFF: one level for now
        scales = [(px_z, px_y, px_x)]  # t, z, y, x
        units = [unit] * 3  # z, y, x

        return {
            "size": [(size_t, size_c, size_z, size_y, size_x)],
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": pixels.attrib.get("Type", "uint16"),
            "axes": "tczyx"
        }

    def _build_dask_array(self) -> List[da.Array]:
        """
        Load pyramid levels from a pyramidal OME-TIFF file, normalize to TCZYX format.

        Returns:
            - List of Dask arrays in TCZYX format
        """
        
        def _reorder_axes(arr: da.Array, axes: str, target_axes: str = "tczyx") -> da.Array:
            """Reorder array to target axes, inserting singleton dims as needed."""
            # Add missing axes
            for ax in target_axes:
                if ax not in axes:
                    arr = da.expand_dims(arr, axis=0)  # prepend singleton dims
                    axes = ax + axes

            # Build permutation order
            permute_order = [axes.index(ax) for ax in target_axes]
            
            return da.transpose(arr, axes=permute_order)
        
        with tifffile.TiffFile(self.path) as tif:
            store = tif.aszarr()
            zgroup = zarr.open(store, mode="r")
            pyramid = [
                    ( 
                        da.from_zarr(zgroup[str(i)]), # image data
                        tif.series[0].levels[i].axes.lower(), # axes order
                    ) for i in range(len(zgroup))
                ]

            data_levels = []
            sizes = []
            scales = []

            # Base pixel sizes (scale)
            base_scale = self.metadata["scales"][0]
            base_size = self.metadata["size"][0][2:] # ZYX only

            for i, level in enumerate(pyramid):

                # Reorder to TCZYX
                arr = _reorder_axes(level[0], level[1])

                # Update scale
                current_size = self.metadata["size"][0][:2] + arr.shape[2:]
                level_scale = (
                                base_scale[0] / current_size[0] * base_size[0],
                                base_scale[1] / current_size[1] * base_size[1],
                                base_scale[2] / current_size[2] * base_size[2],
                                )  # Z, Y, X
                scales.append(level_scale)  # T, C, Z, Y, X
                sizes.append(current_size)
                data_levels.append(arr.rechunk(chunks = self.chunks))

        self.metadata["scales"] = scales
        self.metadata["size"] = sizes

        return data_levels  
    
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
    
