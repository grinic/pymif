import os
import dask.array as da
from tifffile import imread
import zarr
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
from utils.visualize import visualize as _visualize
from utils.write import write as _write
from utils.pyramid import build_pyramid
from dask import delayed

class ViventisManager():
    
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        super().__init__()
        self.position_dir = Path(path)
        self.chunks = chunks
        self.data, self.metadata = self.read()

    def _parse_companion_file(self) -> Dict[str, Any]:
        companion = list(self.position_dir.glob("*.ome"))[0]
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
            "size": (size_t, size_c, size_z, size_y, size_x),
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
        t, c, z, y, x = self.metadata["size"]

        lazy_imread = delayed(imread)  # lazy reader
        filenames = self.metadata["plane_files"]
        lazy_arrays = [ [ lazy_imread(f"{self.position_dir}/{filenames[(ti,ci)]}") for ci in range(c) ] for ti in range(t) ] 
        dask_arrays = [
            [
                da.from_delayed(l2, shape=self.metadata["size"][2:], dtype=self.metadata["dtype"])
                for l2 in l1
            ]
            for l1 in lazy_arrays
        ]
        # Stack into one large dask.array
        stack = da.stack(dask_arrays, axis=0)
        stack = stack.rechunk((self.chunks))
  
        return [stack]  # level 0 only

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        data = self._build_dask_array()
        metadata = self._parse_companion_file()
        return (data, metadata)
    
    def visualize(self, viewer=None):
        return _visualize(self.data, self.metadata, viewer=viewer)

    def write(self, path: str, compressor=None):
        return _write(path, self.data, self.metadata, compressor)
    
    def build_pyramid(self, num_levels: int = 3, downscale_factor: int = 2):
        """
        Converts single-resolution data into a pyramidal multiscale structure
        and updates self.data and self.metadata in-place.
        """
        pyramid_levels, updated_metadata = build_pyramid(
            self.data[0], self.metadata, num_levels=num_levels, downscale_factor=downscale_factor
        )
        self.data = pyramid_levels
        self.metadata = updated_metadata

if __name__ == "__main__":
    test_path = "/g/mif/people/gritti/code/code_ome_zarr/test_data/viventis/20241104_162954_Experiment/Position 1_Settings 1_down"

    if not os.path.exists(test_path):
        print(f"⚠️ Test path not found: {test_path}")
    else:
        import napari
        reader = ViventisManager(test_path)
        data, meta = reader.read()

        print("✅ Metadata keys:", list(meta.keys()))
        print("✅ Axes:", meta["axes"])
        print("✅ Scales:", meta["scales"])
        print("✅ Channel names:", meta["channel_names"])
        
        print(meta)

        # print("✅ Data type:", type(data[0]))
        # print("✅ Data shape:", data.shape)
        # print("✅ Data chunks:", data.chunks)
        
        reader.build_pyramid(3,2)

        print("✅ Metadata keys:", list(meta.keys()))
        print("✅ Axes:", meta["axes"])
        print("✅ Scales:", meta["scales"])
        print("✅ Channel names:", meta["channel_names"])
        
        print(meta)

        reader.write("test.zarr")
        # v = reader.visualize()
        # # v.run()
        
        # napari.run()
        