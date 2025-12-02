import argparse
from argparse import RawTextHelpFormatter
import pymif.microscope_manager as mm
import os
import re
import time
from typing import List, Dict, Any, Optional
from matplotlib.colors import cnames

HEX_PATTERN = re.compile(r'^#?[0-9a-fA-F]{6}$')

def parse_color(value: str) -> str:
    """Parse a CLI color input:
    - Accept 6-digit hex codes (# optional)
    - Accept color names from matplotlib.colors.cnames
    - Raise a meaningful error if invalid
    """

    v = value.strip()

    # --- 1) Hex code (with or without #) ---
    if HEX_PATTERN.match(v):
        return v.replace("#", "").upper()

    # --- 2) Matplotlib named color ---
    lower = v.lower()
    if lower in cnames:
        # cnames returns a hex string with '#', e.g. "#ff00ff"
        return cnames[lower].replace("#", "").upper()

    # --- 3) Fail: report detailed reason ---
    raise argparse.ArgumentTypeError(
        f"Invalid color '{value}'. "
        f"Must be:\n"
        f"  • A 6-digit hex code (e.g. FF00FF or #ff00ff), OR\n"
        f"  • A valid color name from matplotlib ({', '.join(list(cnames.keys())[:10])}, ...)"
    )

def zarr_convert(
        input_path, 
        zarr_path, 
        microscope, 
        max_size : Optional[int] = 100, 
        scene_index : Optional[int] = -1,
        channel_names : Optional[List[str]] = None,
        channel_colors : Optional[List[str]] = None,
        ):

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
