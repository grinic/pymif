from typing import List, Dict, Any, Optional
from pymif.cli.__arguments import _parse_arguments, parse_color
import pymif.microscope_manager as mm
import pandas as pd

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
    else:
        TypeError(f"Microscope {microscope} not recognized. Should be one of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\".")

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
    n_chunks = [2,1,1]
    chunk_size = [
        1, 1, # T, C
        int((dataset.metadata["size"][0][2]/n_chunks[0])+1),  # Z
        int((dataset.metadata["size"][0][3]/n_chunks[1])+1),  # Y
        int((dataset.metadata["size"][0][4]/n_chunks[2])+1)   # X
    ]
    size_mb = 2*chunk_size[2]*chunk_size[3]*chunk_size[4]/1024/1024
    while size_mb > max_size:
        n_chunks = [n_chunks[0]*2,n_chunks[1]*2,n_chunks[2]*2]
        chunk_size = [
            1, 1, # T, C
            int((dataset.metadata["size"][0][2]/n_chunks[0])+1),  # Z
            int((dataset.metadata["size"][0][3]/n_chunks[1])+1),  # Y
            int((dataset.metadata["size"][0][4]/n_chunks[2])+1)   # X
        ]
        size_mb = 2*chunk_size[2]*chunk_size[3]*chunk_size[4]/1024/1024

    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    # --- Initialize manager ---
    if microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=scene_index, chunks=chunk_size)
    else:
        dataset = manager(path=input_path, chunks=chunk_size)

    # --- Build pyramid ---
    print(f"\n--->Selected pyramidal layers, lower layer should have dims<2048")
    n = 1
    shape = [dataset.metadata["size"][0][2], dataset.metadata["size"][0][3], dataset.metadata["size"][0][4]] # [Y, X]
    print(f"Layer {n}, shape {shape}")
    while (shape[0]>2048) or (shape[1]>2048) or (shape[2]>2048):
        n+=1
        shape = [shape[0]//2, shape[1]//2, shape[2]//2]
        print(f"Layer {n}, shape {shape}")

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
        n_ch = dataset.metadata["size"][0][1]
        if len(channel_names)!=n_ch:
            TypeError(f"Length of channel_names={channel_names} does not match dataset channels of length={n_ch}.")
        metadata["channel_names"] = channel_names
        if channel_colors:
            if len(channel_colors)!=n_ch:
                TypeError(f"Length of channel_colors={channel_colors} does not match dataset channels of length={n_ch}.")
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