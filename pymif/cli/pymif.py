from typing import List, Dict, Any, Optional
from pymif.cli.__arguments import _parse_arguments, parse_color
import pymif.microscope_manager as mm
import pandas as pd
import numpy as np


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


def _estimate_levels(metadata):
    axes = _axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    shape = [int(size_map[ax]) for ax in axes if ax in "zyx"]
    if not shape:
        return 1
    n = 1
    print(f"Layer {n}, shape {shape}")
    while any(s > 2048 for s in shape):
        n += 1
        shape = [max(1, s // 2) for s in shape]
        print(f"Layer {n}, shape {shape}")
    return n

def zarr_convert(
        input_path, 
        zarr_path, 
        microscope, 
        max_size : Optional[int] = 100, 
        scene_index : Optional[int] = -1,
        channel_names : Optional[List[str]] = None,
        channel_colors : Optional[List[str]] = None,
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

    """

    if microscope.lower()=="luxendo":
        manager = mm.LuxendoManager
    elif microscope.lower()=="opera":
        manager = mm.OperaManager
    elif microscope.lower()=="viventis":
        manager = mm.ViventisManager
    elif microscope.lower()=="zeiss":
        manager = mm.ZeissManager
    elif microscope.lower()=="zarrv04":
        manager = mm.ZarrV04Manager
    elif microscope.lower()=="zarr":
        manager = mm.ZarrManager
    elif microscope.lower()=='scape':
        manager = mm.ScapeManager
    else:
        TypeError(f"Microscope {microscope} not recognized. Should be one of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\", \"scape\".")

    # --- Figure out chunks dimensions ---
    if microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=scene_index)
    else:
        dataset = manager(path=input_path)
        
    # --- Show metadata summary ---
    print("\n--->Input dataset")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)
    print("DATASET SIZE (MB):", 2*dataset.metadata["size"][0][0]*dataset.metadata["size"][0][1]*dataset.metadata["size"][0][2]*dataset.metadata["size"][0][3]*dataset.metadata["size"][0][4]/1024/1024)

    # --- Select chunk size ---
    print(f"\n--->Select chunks, should not exceed {max_size} MB")
    chunk_size, size_mb, n_chunks = _select_chunk_size(dataset.metadata, max_size)

    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    # --- Initialize manager ---
    if microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=scene_index, chunks=chunk_size)
    else:
        dataset = manager(path=input_path, chunks=chunk_size)

    # --- Build pyramid ---
    print(f"\n--->Selected pyramidal layers, lower layer should have dims<2048")
    n = _estimate_levels(dataset.metadata)

    dataset.build_pyramid(
                        num_levels=n, 
                        downscale_factor=2
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

    # --- Show metadata summary ---
    print("\n--->Input dataset after adjustments")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print(f"CHUNK SIZE: {dataset.chunks} , {size_mb} MB.")
    print(f"N CHUNKS: {n_chunks}.")

    # --- Write to OME-Zarr format ---
    print("\n--->Writing to zarr")
    dataset.to_zarr(zarr_path)

    # --- Show metadata summary for updated dataset ---
    dataset = mm.ZarrManager(path=zarr_path)

def convert_batch(args):
    """Runmode to convert batch of imaged to zarr

    Args:
        args (args): parsed arguments
    """
    cli = f'pymif batch2zarr --input {args.input_file}'
    print(f'Converting batch.\nRunning through: {cli}')

    database = pd.read_csv(args.input_file)

    database = database.fillna("-1")
    print(database)

    for i, v in database.iterrows():
        print("-"*20)
        print(f"Parameters:")

        input_f = v["input"]
        print(f"\tinput_path: {input_f}")

        microscope = v["microscope"]
        print(f"\tmicroscope: {microscope}")

        output = v["output"]
        print(f"\toutput_path: {output}")

        max_size = float(v["max_size(MB)"])
        if max_size==-1:
            max_size = 100
            print(f"\tmax_chunk_size(MB): -1, defaulted to: {max_size}")
        else:
            print(f"\tmax_chunk_size(MB): {max_size}")

        scene_index = int(v["scene_index"])
        if scene_index==-1:
            scene_index = 0
            print(f"\tscene_index: -1, defaulted to: {scene_index}")
        else:
            print(f"\tscene_index: {scene_index}")

        channel_names = v["channel_names"]
        if channel_names=="-1":
            channel_names = None
            print(f"\tchannel_names: -1, defaulted to: {channel_names}")
        else:
            channel_names = [c.strip() for c in channel_names.split(",")]
            print(f"\tchannel_names: {channel_names}")
            
        channel_colors = v["channel_colors"]
        if channel_colors=="-1":
            channel_colors = None
            print(f"\tchannel_colors: -1, defaulted to: {channel_colors}")
        else:
            channel_colors = [c.strip() for c in channel_colors.split(",")]
            channel_colors = [parse_color(c) for c in channel_colors]
            print(f"\tchannel_colors: {channel_colors}")

        zarr_convert(
            input_f, 
            output, 
            microscope,
            max_size,
            scene_index,
            channel_names,
            channel_colors
        )

def convert_single(args):
    """Runmode to convert a single image to zarr

    Args:
        args (args): parsed arguments
    """
    cli = f'pymif 2zarr --input {args.input_path} --zarr_path {args.zarr_path} --microscope {args.microscope} --max_size {args.max_size} --scene_index {args.scene_index} --channel_names {args.channel_names} --channel_colors {args.channel_colors}'
    print(f'Converting single file.\nRunning through: {cli}')
    # Convert 2 zarr
    zarr_convert(
        args.input_path, 
        args.zarr_path, 
        args.microscope,
        args.max_size,
        args.scene_index,
        args.channel_names,
        args.channel_colors
    )

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