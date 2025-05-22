# zarr_converter/core.py

import dask.array as da
from aicsimageio import AICSImage
from aicsimageio.exceptions import UnsupportedFileFormatError
import os
from ome_zarr.scale import multiscale
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from ome_types.model import Metadata, Image, Pixels, Channel, Length

SUPPORTED_EXTS = {".tif", ".tiff", ".czi", ".lif", ".h5", ".hdf5", ".zarr"}

class ZarrConverter:
    """
    Converts microscopy images to pyramidal OME-Zarr format with metadata.

    Attributes:
        input_path (str): Path to the input image file.
        reader (AICSImage): AICSImageIO reader object.
        data (dask.array): Lazy-loaded image data.
    """
    
    def __init__(self, path):
        """
        Initialize the converter and load image.

        Args:
            input_path (str): Path to the input image file (TIF, CZI, LIF, H5).
        """
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise UnsupportedFileFormatError(f"Unsupported file format: {ext}")
        
        self.img = AICSImage(path)
        self.shape = self.img.shape
        self.dims = self.img.dims
        self.pixel_sizes = self.img.physical_pixel_sizes
        self.channel_names = self.img.channel_names or [f"Channel {i}" for i in range(self.shape.C)]

    def get_dask_array(self, level=0, chunks=(1, 1, 1, 256, 256)):
        return self.img.get_dask_data(level=level, chunks=chunks)

    def save_as_ome_zarr(
        self,
        output_path,
        channel_colors=None,
        pixel_size_xy=None,
        pixel_size_z=None,
        downscale_factor=2,
        min_dim=64
    ):
        img_dask = self.get_dask_array()

        if channel_colors is None:
            default_colors = [0x0000FF, 0x00FF00, 0xFF0000, 0xFFFF00]
            channel_colors = [default_colors[i % len(default_colors)] for i in range(self.shape.C)]

        pyramid = multiscale(
            img_dask,
            axes="TCZYX",
            reduction="mean",
            coarsening_xy=downscale_factor,
            min_length=min_dim,
        )

        pixel_size_xy = pixel_size_xy or self.pixel_sizes.X or 0.1
        pixel_size_z = pixel_size_z or self.pixel_sizes.Z or 1.0

        channels = [
            Channel(id=f"Channel:{i}", name=self.channel_names[i], color=channel_colors[i])
            for i in range(self.shape.C)
        ]
        pixels = Pixels(
            dimension_order="TCZYX",
            size_t=self.shape.T,
            size_c=self.shape.C,
            size_z=self.shape.Z,
            size_y=self.shape.Y,
            size_x=self.shape.X,
            type=str(img_dask.dtype),
            channels=channels,
            physical_size_x=Length(value=pixel_size_xy, unit="micrometer"),
            physical_size_y=Length(value=pixel_size_xy, unit="micrometer"),
            physical_size_z=Length(value=pixel_size_z, unit="micrometer"),
        )
        metadata = Metadata(images=[Image(id="Image:0", name="PyramidImage", pixels=pixels)])

        store = parse_url(output_path, mode="w").store
        write_image(
            image=pyramid,
            group=store,
            axes="TCZYX",
            multiscales=True,
            metadata=metadata,
        )

    def view(self, level=0):
        try:
            import napari
            data = self.get_dask_array(level=level).squeeze()
            napari.view_image(data, channel_axis=0, name=self.channel_names)
        except ImportError:
            print("Napari is not installed. Run `pip install napari[all]` to enable viewer.")


