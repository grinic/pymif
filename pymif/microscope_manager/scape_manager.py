import dask.array as da
from dask import delayed
from tifffile import imread, TiffFile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .microscope_manager import MicroscopeManager


class ScapeManager(MicroscopeManager):
    """
    MicroscopeManager for SCAPE microscope datasets where metadata is stored in an external
    .xlif file located in a 'Metadata' folder next to the OME-TIFF.
    """

    def __init__(
        self,
        ome_tiff_path: str,
        chunks: Tuple[int, ...] = (1, 1, 8, 1024, 1024),
    ):
        super().__init__()
        self.ome_tiff_path = Path(ome_tiff_path)
        self.chunks = chunks
        self.read()

    # ---------- Path resolution helpers ----------

    @staticmethod
    def _strip_ome_tiff_suffix(filename: str) -> str:
        """
        Convert typical Leica names like:
          'p1 (2).ome.tif'  -> 'p1 (2)'
          'p1 (2).ome.tiff' -> 'p1 (2)'
          'p1 (2).tif'      -> 'p1 (2)'
        """
        lower = filename.lower()
        if lower.endswith(".ome.tiff"):
            return filename[:-len(".ome.tiff")]
        if lower.endswith(".ome.tif"):
            return filename[:-len(".ome.tif")]
        if lower.endswith(".tiff"):
            return filename[:-len(".tiff")]
        if lower.endswith(".tif"):
            return filename[:-len(".tif")]
        # Fallback: drop only the last suffix
        return Path(filename).stem

    def _find_xlif_for_ome_tiff(self) -> Path:
        """
        Given an OME-TIFF path, look for:
          <ome_dir>/Metadata/<base_name>.xlif

        If not found, fall back to searching for any .xlif inside <ome_dir>/Metadata.
        """
        if not self.ome_tiff_path.is_file():
            raise FileNotFoundError(f"OME-TIFF file not found: {self.ome_tiff_path}")

        ome_dir = self.ome_tiff_path.parent
        metadata_dir = ome_dir / "Metadata"
        if not metadata_dir.exists() or not metadata_dir.is_dir():
            raise FileNotFoundError(f"'Metadata' directory not found next to OME-TIFF: {metadata_dir}")

        base_name = self._strip_ome_tiff_suffix(self.ome_tiff_path.name)
        expected = metadata_dir / f"{base_name}.xlif"
        if expected.exists():
            return expected

        # Fallback: if naming is slightly different, try to find any .xlif
        xlifs = sorted(metadata_dir.glob("*.xlif"))
        if not xlifs:
            raise FileNotFoundError(
                f"No .xlif files found in: {metadata_dir} "
                f"(expected: {expected.name})"
            )

        # Prefer files that contain the base_name (case-insensitive)
        base_lower = base_name.lower()
        for x in xlifs:
            if base_lower in x.name.lower():
                return x

        # Otherwise, return the first .xlif found
        return xlifs[0]

    def _convert_spatial_units_to_micrometers(self):
        """
        Convert spatial scales (Z, Y, X) from meters to micrometers (µm)
        if they are currently stored in meters.

        Updates:
            self.metadata["scales"]
            self.metadata["units"]
        """

        if "scales" not in self.metadata or "units" not in self.metadata:
            raise ValueError("Metadata must contain 'scales' and 'units' fields.")

        scales = list(self.metadata["scales"][0])
        units = list(self.metadata["units"])

        converted_scales = []
        converted_units = []

        for scale, unit in zip(scales, units):

            if unit in ("m", "meter", "meters"):
                converted_scales.append(scale * 1e6)   # meters → micrometers
                converted_units.append("µm")

            elif unit in ("mm",):
                converted_scales.append(scale * 1e3)   # mm → µm
                converted_units.append("µm")

            elif unit in ("µm", "um"):
                converted_scales.append(scale)
                converted_units.append("µm")

            else:
                # Leave unchanged if unknown
                converted_scales.append(scale)
                converted_units.append(unit)

        self.metadata["scales"] = [tuple(converted_scales)]
        self.metadata["units"] = tuple(converted_units)

    # ---------- Metadata parsing ----------

    def _parse_xlif_metadata(self) -> Dict[str, Any]:
        """Parse dimensional, physical, and channel metadata from the matched .xlif file."""
        xlif_path = self._find_xlif_for_ome_tiff()
        tree = ET.parse(xlif_path)
        root = tree.getroot()

        imgdesc = root.find(".//ImageDescription")
        if imgdesc is None:
            raise ValueError("<ImageDescription> element not found in .xlif file")

        dims_node = imgdesc.find("Dimensions")
        if dims_node is None:
            raise ValueError("<Dimensions> element not found in .xlif file")

        dim_descs = dims_node.findall("DimensionDescription")
        dim_by_id = {int(d.attrib["DimID"]): d.attrib for d in dim_descs}

        # Leica/LAS AF convention (based on your example):
        # DimID=1 -> X, DimID=2 -> Y, DimID=3 -> Z
        size_x = int(dim_by_id[1]["NumberOfElements"])
        size_y = int(dim_by_id[2]["NumberOfElements"])
        size_z = int(dim_by_id[3]["NumberOfElements"])
        size_t = int(dim_by_id[4]["NumberOfElements"])

        len_x = float(dim_by_id[1]["Length"])
        len_y = float(dim_by_id[2]["Length"])
        len_z = float(dim_by_id[3]["Length"])
        len_t = float(dim_by_id[4]["Length"])

        unit_x = dim_by_id[1].get("Unit", "m")
        unit_y = dim_by_id[2].get("Unit", "m")
        unit_z = dim_by_id[3].get("Unit", "m")
        unit_t = dim_by_id[4].get("Unit", "s")

        # Channel information
        channels_node = imgdesc.find("Channels")
        channel_descs = (
            channels_node.findall("ChannelDescription")
            if channels_node is not None else []
        )
        size_c = max(1, len(channel_descs))

        channel_names = []
        channel_colors = []

        lut_to_hex = {
            "Cyan": "#00FFFF",
            "Magenta": "#FF00FF",
            "Yellow": "#FFFF00",
            "Red": "#FF0000",
            "Green": "#00FF00",
            "Blue": "#0000FF",
            "Gray": "#808080",
            "Grey": "#808080",
            "White": "#FFFFFF",
        }

        for i, ch in enumerate(channel_descs):
            lut = ch.attrib.get("LUTName", "").strip()
            channel_names.append(lut if lut else f"Channel {i}")
            channel_colors.append(lut_to_hex.get(lut, "#FFFFFF"))

        dtype = "uint16"

        # Compute voxel spacing from physical length
        scale_x = len_x / size_x if size_x else 1.0
        scale_y = len_y / size_y if size_y else 1.0
        scale_z = len_z / size_z if size_z else 1.0
        scale_t = len_t / size_t if size_t else 1.0

        return {
            "size": [(size_t, size_c, size_z, size_y, size_x)],
            "scales": [(scale_z, scale_y, scale_x)],
            "units": (unit_z, unit_y, unit_x),
            "time_increment": 1.0,
            "time_increment_unit": unit_t,
            "channel_names": (
                channel_names
                if channel_names
                else [f"Channel {i}" for i in range(size_c)]
            ),
            "channel_colors": (
                channel_colors
                if channel_colors
                else ["#FFFFFF"] * size_c
            ),
            "dtype": dtype,
            "axes": "tczyx",
            "xlif_path": str(xlif_path),
            "ome_tiff_path": str(self.ome_tiff_path.resolve()),
        }

    # ---------- Dask array construction ----------

    def _build_dask_array(self) -> List[da.Array]:
        """Build a TCZYX dask array from the provided OME-TIFF file."""
        ome_path = self.ome_tiff_path.resolve()

        with TiffFile(str(ome_path)) as tf:
            series = tf.series[0]
            tif_shape = series.shape
            tif_axes = getattr(series, "axes", None)

        lazy_read = delayed(imread)
        arr = da.from_delayed(
            lazy_read(str(ome_path)),
            shape=tif_shape,
            dtype=self.metadata["dtype"],
        )

        if tif_axes is None:
            # Fallback if axes are not defined
            t, c, z, y, x = self.metadata["size"][0]
            if arr.ndim == 3:      # (z, y, x)
                arr = arr[None, None, ...]
            elif arr.ndim == 4:    # (z, c, y, x) or (c, z, y, x)
                if arr.shape[1] == c:
                    arr = arr[None, ...]
                    arr = arr.transpose(0, 2, 1, 3, 4)
                elif arr.shape[0] == c:
                    arr = arr[None, ...]
                else:
                    arr = arr[None, None, ...]
        else:
            axes_in = tif_axes.upper()
            target = "TCZYX"
            idx = {ax: i for i, ax in enumerate(axes_in)}

            # Add missing singleton dimensions
            for ax in target:
                if ax not in idx:
                    arr = da.expand_dims(arr, axis=0)
                    axes_in = ax + axes_in
                    idx = {a: i for i, a in enumerate(axes_in)}

            perm = [idx[ax] for ax in target]
            arr = arr.transpose(*perm)

        arr = arr.rechunk(self.chunks)
        return [arr]

    def read(self):
        """Main entry point: parse metadata and build data array."""
        self.metadata = self._parse_xlif_metadata()
        self._convert_spatial_units_to_micrometers()
        self.data = self._build_dask_array()
        return
