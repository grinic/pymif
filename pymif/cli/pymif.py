from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pymif.microscope_manager as mm
from pymif.cli.__arguments import _parse_arguments, parse_color, parse_downscale_factor


def _axes(metadata):
    return str(metadata.get("axes", "tczyx")).lower()


def _axis_size(metadata, axis, default=1):
    axes = _axes(metadata)
    return int(metadata["size"][0][axes.index(axis)]) if axis in axes else default


def _dataset_size_mb(metadata):
    dtype = np.dtype(metadata.get("dtype", "uint16"))
    nbytes = dtype.itemsize
    for size in metadata["size"][0]:
        nbytes *= int(size)
    return nbytes / 1024 / 1024


def _select_chunk_size(metadata, max_size):
    axes = _axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    n_chunks = {"z": 2, "y": 1, "x": 1}
    chunk_map = {"t": 1, "c": 1}
    while True:
        chunk_map.update({
            "z": int(size_map.get("z", 1) / n_chunks["z"] + 1),
            "y": int(size_map.get("y", 1) / n_chunks["y"] + 1),
            "x": int(size_map.get("x", 1) / n_chunks["x"] + 1),
        })
        dtype_size = np.dtype(metadata.get("dtype", "uint16")).itemsize
        size_mb = dtype_size * chunk_map["z"] * chunk_map["y"] * chunk_map["x"] / 1024 / 1024
        if size_mb <= max_size:
            break
        n_chunks = {key: value * 2 for key, value in n_chunks.items()}
    chunks = tuple(int(max(1, min(size_map.get(ax, 1), chunk_map[ax]))) for ax in axes)
    return chunks, size_mb, n_chunks


def _estimate_levels(metadata, downscale_factor):
    axes = _axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    shape = [int(size_map[ax]) for ax in axes if ax in "zyx"]
    if not shape:
        return 1
    n = 1
    print(f"Layer {n}, shape {shape}")
    while any(s > 2048 for s in shape):
        n += 1
        shape = [max(1, s // downscale_factor[i]) for i, s in enumerate(shape)]
        print(f"Layer {n}, shape {shape}")
    return n


def _normalize_downscale_factor(value):
    if value is None:
        return 2
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        return tuple(int(v) for v in value)
    return int(value)


def _present(value):
    return not (
        pd.isna(value)
        or (isinstance(value, str) and value.strip() == "")
        or value == "-1"
    )

def _manager_name_map() -> Dict[str, Any]:
    return {
        'luxendo': mm.LuxendoManager,
        'opera': mm.OperaManager,
        'viventis': mm.ViventisManager,
        'zeiss': mm.ZeissManager,
        'zarrv04': mm.ZarrV04Manager,
        'zarr': mm.ZarrManager,
        'scape': mm.ScapeManager,
    }

def _resolve_zarr_manager(input_path: str, microscope: Optional[str] = None):
    """Resolve a manager class from an explicit microscope name or by inspecting the path."""

    if _present(microscope):
        key = str(microscope).strip().lower()
        manager = _manager_name_map().get(key)
        if manager is None:
            raise TypeError(
                f'Microscope {microscope} not recognized. Should be one of '
                f'"luxendo", "opera", "viventis", "zeiss", "zarrv04", "zarr", "scape".'
            )
        return manager, key

    path = Path(input_path)

    # Zarr stores: prefer explicit v3/v0.5 when zarr.json exists, otherwise legacy v0.4.
    if path.is_dir():
        if (path / 'zarr.json').exists():
            return mm.ZarrManager, 'zarr'
        if (path / '.zgroup').exists() or (path / '.zattrs').exists():
            return mm.ZarrManager, 'zarrv04'

        if any(path.glob('*.lux.h5')) and any(path.glob('*.xml')):
            return mm.LuxendoManager, 'luxendo'

        if any(path.glob('*.ome')) and any(path.glob('*.tif')):
            return mm.ViventisManager, 'viventis'

    # File-based formats.
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.czi'):
        return mm.ZeissManager, 'zeiss'

    if suffixes.endswith('.ome.tif') or suffixes.endswith('.ome.tiff') or path.suffix.lower() in {'.tif', '.tiff'}:
        metadata_dir = path.parent / 'Metadata'
        if metadata_dir.exists() and any(metadata_dir.glob('*.xlif')):
            return mm.ScapeManager, 'scape'
        return mm.OperaManager, 'opera'

    raise TypeError(
        'Could not auto-detect the microscope from the input path. '
        'Pass --microscope explicitly with one of "luxendo", "opera", "viventis", "zeiss", "zarrv04", "zarr", "scape".'
    )

def zarr_convert(
    input_path, 
    zarr_path, 
    microscope: Optional[str] = None, 
    max_size : Optional[int] = 100, 
    chunk_size : Optional[List[int]] = None,
    scene_index : Optional[int] = 1,
    channel_names : Optional[List[str]] = None,
    channel_colors : Optional[List[str]] = None,
    zarr_format : Optional[int] = 3,
    downscale_factor: Optional[int] = 2,
    num_levels: Optional[int] = None,
):
    """Helper function for CLI to convert a dataset to zarr given some parameters.

    Parameters
    ----------
        input_path : str
            Input path for the data to be converted.
        zarr_path : str
            Output .zarr path.
        microscope : str
            Microscope used to acquire input data.
            One of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\".
        max_size : Optional[int]
            Max chunk size in MB. \n
            Default: 100
        chunk_size : Optional[List[int]]
            Chunk size in TCZYX format, or whatever axes are present in the dataset.
            Default: None
        scene_index : Optional[int]
            Scene index for .czi files. \n
            Default: -1
        channel_names : Optional[List[str]]
            Name of channels.\n
            Example: \"-cn bf gfp rfp\"\n
            Default: None
        channel_colors : Optional[List[str]]
            Colors of channels (hex or matplotlib color name)\n
            Example: \"-cc 0000FF cyan 00ff00\")\n
            Default: None
        zarr_format : Optional[int]
            Output zarr format. Zarr v2 maps to NGFF 0.4 and Zarr v3 maps to NGFF 0.5.\n
            Default: 3
        downscale_factor : Optional[int]
            Pyramid downsampling factor.\n
            Example: \"-df 1 2 2\")\n
            Default: 2
        num_levels : Optional[int]
            Number of pyramid levels.\n
            Example: \"-nl 3\")\n
            Default: None
    """

    manager, resolved_microscope = _resolve_zarr_manager(input_path, microscope)
    print(f'\n--->Using manager: {manager.__name__} ({resolved_microscope})')

    downscale_factor = _normalize_downscale_factor(downscale_factor)
    
    # --- Figure out chunks dimensions ---
    if resolved_microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=scene_index)
    else:
        dataset = manager(path=input_path)
        
    # --- Show metadata summary ---
    print("\n--->Input dataset")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)
    print("DATASET SIZE (MB):", _dataset_size_mb(dataset.metadata))

    # --- Select chunk size ---
    print(f"\n--->Select chunks.")
    if chunk_size is not None:
        print(f"\tUsing user-provided chunk size: {chunk_size}")
        size_mb = _dataset_size_mb(dataset.metadata) * np.prod(chunk_size) / np.prod(dataset.metadata["size"][0])
        n_chunks = {ax: int(np.ceil(size / chunk)) for ax, size, chunk in zip(_axes(dataset.metadata), dataset.metadata["size"][0], chunk_size)}
    else:
        print(f"\tUsing max_size={max_size} MB to select chunk size.")
        chunk_size, size_mb, n_chunks = _select_chunk_size(dataset.metadata, max_size)

    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    # --- Initialize manager ---
    if resolved_microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=scene_index, chunks=chunk_size)
    else:
        dataset = manager(path=input_path, chunks=chunk_size)

    # --- Build pyramid ---
    print(f"\n--->Selected pyramidal layers, lower layer should have dims<2048")
    if num_levels is None:
        num_levels = _estimate_levels(dataset.metadata, downscale_factor)

    dataset.build_pyramid(
        num_levels=num_levels, 
        downscale_factor=downscale_factor
    )

    # --- Modify metadata according to optional parameters ---
    
    # Metadata format:
    # metadata = {
    #         # "size": [(size_t, size_c, size_z, size_y, size_x)], # can't change
    #         # "scales": scales, # can't change
    #         # "units": units, # can't change
    #         # "time_increment": time_increment, # can't change
    #         # "time_increment_unit": time_unit, # can't change
    #         "channel_names": channel_names,
    #         "channel_colors": channel_colors,
    #         # "dtype": pixels.attrib.get("Type", "uint16"), # can't change
    #         # "axes": "tczyx" # can't change
    #     }

    print("\n--->Updating metadata to selected channel_names and channel_colors")
    metadata = {}
    if channel_names:
        if "c" not in _axes(dataset.metadata):
            raise TypeError("channel_names were provided, but the dataset has no channel axis.")
        n_ch = _axis_size(dataset.metadata, "c")
        if len(channel_names)!=n_ch:
            raise TypeError(f"Length of channel_names={channel_names} does not match dataset channels of length={n_ch}.")
        metadata["channel_names"] = channel_names
        if channel_colors:
            if len(channel_colors)!=n_ch:
                raise TypeError(f"Length of channel_colors={channel_colors} does not match dataset channels of length={n_ch}.")
            metadata["channel_colors"] = channel_colors
    dataset.update_metadata(metadata)

    print("\n--->Updating metadata to selected zarr_format and downscale_factor")
    ngff_version = '0.4' if int(zarr_format) == 2 else '0.5'

    # --- Show metadata summary ---
    print("\n--->Input dataset after adjustments")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print(f"CHUNK SIZE: {dataset.chunks} , {size_mb} MB.")
    print(f"N CHUNKS: {n_chunks}.")
    print(f"PYRAMID LEVELS: {num_levels}.")
    print(f"ZARR FORMAT: {zarr_format}, NGFF VERSION: {ngff_version}.")

    # --- Write to OME-Zarr format ---
    print("\n--->Writing to zarr")
    dataset.to_zarr(zarr_path, zarr_format=int(zarr_format), ngff_version=ngff_version)

    # --- Show metadata summary for updated dataset ---
    dataset = mm.ZarrManager(path=zarr_path)

def convert_batch(args):
    """Runmode to convert batch of imaged to zarr

    Args:
        args (args): parsed arguments
    """
    cli = f"pymif batch2zarr --input {args.input_file}"
    print(f"Converting batch.\nRunning through: {cli}")

    database = pd.read_csv(args.input_file)
    print(database)

    for i, v in database.iterrows():
        print("-" * 20)

        input_f = v["input"]
        output = v["output"]
        microscope = v["microscope"]

        conv_kwargs = {
            "input_path": input_f,
            "zarr_path": output,
            "microscope": microscope,
        }

        if "max_size(MB)" in database.columns and _present(v.get("max_size(MB)")):
            conv_kwargs["max_size"] = float(v["max_size(MB)"])

        # Prioritize chunk_size over max_size if both are present
        if "chunk_size" in database.columns and _present(v.get("chunk_size")):
            conv_kwargs["chunk_size"] = parse_downscale_factor(v["chunk_size"])
            conv_kwargs.pop("max_size", None)
        else:
            if "max_size(MB)" in database.columns and _present(v.get("max_size(MB)")):
                conv_kwargs["max_size"] = float(v["max_size(MB)"])

        if "scene_index" in database.columns and _present(v.get("scene_index")):
            conv_kwargs["scene_index"] = int(v["scene_index"])

        if "channel_names" in database.columns and _present(v.get("channel_names")):
            conv_kwargs["channel_names"] = [c.strip() for c in str(v["channel_names"]).split(",")]

        if "channel_colors" in database.columns and _present(v.get("channel_colors")):
            conv_kwargs["channel_colors"] = [
                parse_color(c.strip()) for c in str(v["channel_colors"]).split(",")
            ]

        if "zarr_format" in database.columns and _present(v.get("zarr_format")):
            conv_kwargs["zarr_format"] = int(v["zarr_format"])

        if "downscale_factor" in database.columns and _present(v.get("downscale_factor")):
            conv_kwargs["downscale_factor"] = parse_downscale_factor(v["downscale_factor"])

        if "num_levels" in database.columns and _present(v.get("num_levels")):
            conv_kwargs["num_levels"] = int(v["num_levels"])

        zarr_convert(**conv_kwargs)

def convert_single(args):
    """Runmode to convert a single image to zarr

    Args:
        args (args): parsed arguments
    """
    cli = (
        f'pymif 2zarr --input {args.input_path} --zarr_path {args.zarr_path} '
        f'--microscope {args.microscope} --max_size {args.max_size} '
        f'--scene_index {args.scene_index} --channel_names {args.channel_names} '
        f'--channel_colors {args.channel_colors} --zarr_format {args.zarr_format} '
        f'--num_levels {args.num_levels} --downscale_factor {args.downscale_factor} '
        f'--chunk_size {args.chunk_size}'
    )
    print(f'Converting single file.\nRunning through: {cli}')
    exclude = {"runmode"}
    conv_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in exclude and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "chunk_size" in conv_kwargs:
        conv_kwargs.pop("max_size", None)

    # Convert 2 zarr
    zarr_convert(**conv_kwargs)

def main():
    """Main fxn

        Here the PyMIF will decide which runmode to execute based on the positional argument after pymif command.
    """
    args = _parse_arguments()
    if args.runmode == 0:
        convert_single(args)
    elif args.runmode == 1:
        convert_batch(args)
    # TODO There is room for more runmodes possibly in the future

if __name__ == "__main__":
    main()