# readers/zeiss_reader.py
import tempfile
import os
import requests
from aicsimageio import AICSImage
import dask.array as da
from pathlib import Path
from typing import List, Tuple

class ZeissReader:
    def __init__(self, 
                 file_path: Path,
                 scene_idx: int = None):
        self.file_path = Path(file_path)
        self.img = AICSImage(self.file_path)
        
        if (self.img.scenes>1)and(scene_idx==None):
            
            raise(EnvironmentError("Importing multiple scenes is not supported. Please select a scene."))

    def read(self) -> [List[da.Array], List[dict]]:
        """Return one (image, metadata) tuple per scene, each with one resolution layer."""
        scene_outputs = []

        for scene in self.img.scenes:
            self.img.set_scene(scene)
            data = self.img.get_image_dask_data("TCZYX")
            metadata = self._build_metadata(scene)
            scene_outputs.append(([data], [metadata]))

        return scene_outputs

    def _build_metadata(self, scene: str) -> dict:
        ome_md = self.img.ome_metadata
        print(ome_md)

        # Pixel size and units
        pixel_size = {
            "X": self.img.physical_pixel_sizes.X if self.img.physical_pixel_sizes else 1.,
            "Y": self.img.physical_pixel_sizes.Y if self.img.physical_pixel_sizes else 1.,
            "Z": self.img.physical_pixel_sizes.Z if self.img.physical_pixel_sizes else 1.,
        }
        print(pixel_size)

        pixel_units = str(self.img.physical_pixel_units) if self.img.physical_pixel_sizes and self.img.physical_pixel_units else "pixel"
        print(pixel_units)

        # Channel info
        channel_names = []
        channel_colors = []
        exposure_times = []

        if hasattr(ome_md, "channels"):
            for ch in md.channels:
                channel_names.append(getattr(ch, "name", None))

                # Color
                color = getattr(ch, "color", None)
                color_hex = f"#{color & 0xFFFFFF:06x}" if color else None
                channel_colors.append(color_hex)

                # Exposure time
                exp_time = getattr(ch, "exposure_time", None)
                exposure_times.append(exp_time)

        # Laser info (may be missing or incomplete)
        laser_lines = []
        laser_power = {}

        if hasattr(md, "instrument") and md.instrument and hasattr(md.instrument, "light_sources"):
            for laser in md.instrument.light_sources:
                wl = getattr(laser, "wavelength", None)
                power = getattr(laser, "power", None)

                if wl:
                    wl_val = wl.value if hasattr(wl, "value") else wl
                    laser_lines.append(wl_val)

                    if power:
                        power_val = power.value if hasattr(power, "value") else power
                        laser_power[wl_val] = power_val

        return {
            "scene_name": scene,
            "dimensions": self.img.dims,
            "shape": self.img.shape,
            "pixel_size": pixel_size,
            "pixel_size_units": pixel_units,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "exposure_times": exposure_times,
            "laser_lines": laser_lines,
            "laser_power": laser_power,
            "raw_metadata": md,
        }
    
    def save_as_ome_zarr(self, output_dir: Path, scale_factors=[2, 2]):
        """Save as pyramidal OME-Zarr with metadata"""
        from ome_zarr.writer import write_image
        data = self.read()
        axes = "CZYX"
        write_image(output_dir, data, axes=axes, scale_factors=scale_factors, metadata=self.get_metadata())


def test_ZeissReader():
    url = "https://zenodo.org/record/7015307/files/S=2_T=3_Z=5_CH=1.czi?download=1"
    
    with tempfile.NamedTemporaryFile(suffix=".czi", delete=True) as tmp:
        print(f"Downloading test CZI file to: {tmp.name}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.flush()

        # Import your ZeissReader from the current module
        reader = ZeissReader(tmp.name)
        results = reader.read()

        assert isinstance(results, list), "Reader should return a list of scene tuples"
        assert len(results) == 2, "Expected 2 scenes in test file"

        for i, (arrays, metadatas) in enumerate(results):
            print(f"Scene {i} — Resolutions: {len(arrays)}")
            assert isinstance(arrays[0], da.Array), "Output should be a Dask array"
            assert isinstance(metadatas[0], dict), "Metadata should be a dictionary"
            assert "pixel_size" in metadatas[0], "Metadata should contain pixel size"
            assert "scene_name" in metadatas[0], "Metadata should contain scene name"
            assert arrays[0].shape[:3] == (3, 1, 5), "Expected shape T=3, C=1, Z=5"

        print("✅ ZeissReader test passed successfully.")

# Run test if this module is executed directly
if __name__ == "__main__":
    test_ZeissReader()