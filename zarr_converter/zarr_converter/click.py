import click
from .core import ZarrConverter

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--pixel-size-xy", type=float, default=None, help="Pixel size in XY (microns).")
@click.option("--pixel-size-z", type=float, default=None, help="Pixel size in Z (microns).")
@click.option("--downscale-factor", type=int, default=2, help="Downscale factor for pyramid.")
@click.option("--min-dim", type=int, default=64, help="Minimum dimension size for pyramid.")
def convert(input_path, output_path, pixel_size_xy, pixel_size_z, downscale_factor, min_dim):
    """
    Convert INPUT_PATH image (TIFF, CZI, LIF, H5) to pyramidal OME-Zarr at OUTPUT_PATH.
    """
    converter = ZarrConverter(input_path)
    converter.save_as_ome_zarr(
        output_path,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        downscale_factor=downscale_factor,
        min_dim=min_dim
    )
    print(f"Converted to: {output_path}")

if __name__ == "__main__":
    convert()
