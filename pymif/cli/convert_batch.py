"""
Command-line interface for the pymif package.
To run:
>> conda activate pymif
(pymif) >> ./convert_batch.py -i <input>
"""

import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
# import os
# import time
from .auto_zarr_convert import zarr_convert, parse_color

description = "Command-line interface for the pymif package to batch convert several datasets to zarr.\n\n"\
                "To run:\n"\
                ">>> conda activate pymif\n"\
                "(pymif) >>> pymif-batch2zarr -i INPUT_FILE\n\n"\
                "The INPUT_FILE is a .csv file of the form:\n\n"\
                " input   | microscope | output       | max_size(MB) | scene_index | channel_colors | channel_names \n"\
                " /path/1 | opera      | /path/1.zarr |              |             | lime,white     | gfp,bf        \n"\
                " /path/2 | viventis   | /path/2.zarr | 1000         | 0           | 0000FF, FF00FF |               \n"\
                " /path/3 | luxendo    | /path/4.zarr |              |             | lime, ff00ff   |               \n"\
                " /path/4 | zeiss      | /path/5.zarr | 100          | 1           |                |               \n\n"\
                "All column headers are mandatory, but values can be empty\n"\
                "channel_colors can be hex code or valid matplotlib colors.\n"\
                "An example .csv file can be found in \"pymif/examples\" folder."\

def main():
    """Command-line interface for the pymif package to batch convert several datasets to zarr.

    More info with:

    >>> pymif-batch2zarr --help
    """
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("--input_file", "-i", required=True, help="Path to the input .csv file.")

    args = parser.parse_args()

    print(f"Running with: {args}")

    database = pd.read_csv(args.input_file)
    print(database)

    database = database.fillna("-1")
    print(database)

    for i, v in database.iterrows():

        print("-"*20)
        print(f"Parameters:")

        input = v["input"]
        print(f"\tinput_path: {input}")

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
        # print(
            input, 
            output, 
            microscope,
            max_size,
            scene_index,
            channel_names,
            channel_colors
            )

if __name__ == "__main__":
    main()
