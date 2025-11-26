#!/usr/bin/env python3

###
# To run:
# >> conda activate pymif
# (pymif) >> ./convert_batch.py -i <input>
###

import argparse
import pymif.microscope_manager as mm
import pandas as pd
import os
import time

def zarr_convert(input_path, zarr_path, microscope, max_size):
    print("-"*20)
    print(f"Converting data:\n input_path: {input_path}\n microscope: {microscope}\n output_path: {zarr_path}\n max_chunk_size(MB): {max_size}")

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

    dataset = manager(path=input_path)
    # --- Show metadata summary ---
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("DATASET SIZE (GB):", 2*dataset.metadata["size"][0][0]*dataset.metadata["size"][0][1]*dataset.metadata["size"][0][2]*dataset.metadata["size"][0][3]*dataset.metadata["size"][0][4]/1024/1024/1024)

    # --- Figure out chunks dimensions ---
    # make sure each chunk does not exceed the specified size in MB
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
    print(f"CHUNKS DIMS (TCZYX): {chunk_size}")
    print(f"CHUNKS SIZE (MB): {size_mb}")
    print(f"N CHUNKS: {n_chunks}")

    # --- Initialize manager ---
    dataset = manager(path=input_path, chunks=chunk_size)

    # --- Build pyramid if not already ---
    # make sure last layer has no dimension exceedinf 2048 (MAX_GL for 3D rendering)
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
        description="Process an image with scaling."
    )

    parser.add_argument("--input_file", "-i", required=True, help="Path to the input file.")

    args = parser.parse_args()

    print(f"Running with: {args}")

    lines = []
    for line in open(args.input_file).readlines():
        lines.append(line.strip().split())

    d = {}
    keys = lines[0]
    for k in keys:
        d[k] = []
    for a in lines[1:]:
        for k, b in zip(keys, a):
            d[k].append(b)
    d = pd.DataFrame(d)
    # print(d)

    for i, v in d.iterrows():
        # print("\n")
        # print(v["input"])
        zarr_convert(
            v["input"], 
            v["output"], 
            v["microscope"],
            float(v["max_size(MB)"]),
            )


if __name__ == "__main__":
    main()
