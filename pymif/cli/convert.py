"""
Command-line interface for the pymif package to convert a single dataset
To run:
>> conda activate pymif
(pymif) >> ./convert_cli.py -i <input> -z <zarr> -m <microscope>
"""

import argparse
from argparse import RawTextHelpFormatter
import pymif.microscope_manager as mm
import os
import time

def zarr_convert(input_path, zarr_path, microscope, max_size, scene_index=-1):

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

    # --- Figure out chunks dimensions ---
    if microscope.lower() == "zeiss":
        dataset = manager(path=input_path,scene_index=scene_index)
    else:
        dataset = manager(path=input_path)
    # --- Show metadata summary ---
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)
    print("DATASET SIZE (MB):", 2*dataset.metadata["size"][0][0]*dataset.metadata["size"][0][1]*dataset.metadata["size"][0][2]*dataset.metadata["size"][0][3]*dataset.metadata["size"][0][4]/1024/1024)

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

    print("\n")
    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    # --- Initialize manager ---
    if microscope.lower() == "zeiss":
        dataset = manager(path=input_path,scene_index=scene_index, chunks=chunk_size)
    else:
        dataset = manager(path=input_path, chunks=chunk_size)

    # --- Show metadata summary ---
    print("\n")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)

    # --- Build pyramid if not already ---
    n = 1
    shape = [dataset.metadata["size"][0][2], dataset.metadata["size"][0][3], dataset.metadata["size"][0][4]] # [Y, X]
    print("\n")
    print(f"Layer {n}, shape {shape}")
    while (shape[0]>2048) or (shape[1]>2048) or (shape[2]>2048):
        n+=1
        shape = [shape[0]//2, shape[1]//2, shape[2]//2]
        print(f"Layer {n}, shape {shape}")

    print("\n")
    dataset.build_pyramid(
                        num_levels=n, 
                        downscale_factor=2
                        )

    # --- Write to OME-Zarr format ---
    dataset.to_zarr(zarr_path)

    # --- Show metadata summary for updated dataset ---
    dataset = mm.ZarrManager(path=zarr_path)

def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for the pymif package to convert a single dataset in zarr.\n\
                    To run:\n\
                    >> conda activate pymif\n\
                    (pymif) >> pymif-2zarr -i <input> -z <zarr> -m <microscope> -ms <max_size> -si <scene_index>",
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("--input_path", "-i", required=True, help="Path to the input file.")
    parser.add_argument("--zarr_path", "-z", required=True, help="Path to the output zarr.")
    parser.add_argument("--microscope", "-m", required=True, 
                        help="Microscope. One of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\".")
    parser.add_argument("--max_size", "-ms", required=False, default=100, help="Max chunk size in MB.")
    parser.add_argument("--scene_index", "-si", required=False, default=0, help="Scene index for .czi files.")

    args = parser.parse_args()

    print(f"Running with: {args}")

    zarr_convert(
        args.input_path, 
        args.zarr_path, 
        args.microscope,
        float(args.max_size),
        int(args.scene_index),
        )


if __name__ == "__main__":
    main()
