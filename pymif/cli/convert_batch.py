"""
Command-line interface for the pymif package.
To run:
>> conda activate pymif
(pymif) >> ./convert_batch.py -i <input>
"""

import argparse
from argparse import RawTextHelpFormatter
import pymif.microscope_manager as mm
import pandas as pd
import os
import time
from auto_zarr_convert import zarr_convert, parse_color

def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for the pymif package.\n\n"
                    "To run:\n"
                    ">> conda activate pymif\n"
                    "(pymif) >> pymif-batch2zarr -i <input>\n"
                    "The <input> .csv file should be of the form:\n\n"
                    "input      microscope      output            max_size(MB)    scene_index     channel_colors      channel_names   \n"
                    "/path/1    opera           /path/1.zarr      100             0               lime,white          gfp,bf          \n"
                    "/path/2    viventis        /path/2.zarr      100             0               0000FF, FF00FF                      \n"
                    "/path/3    zeiss           /path/3.zarr      100             1                                                   \n",
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("--input_file", "-i", required=True, help="Path to the input .csv file.")

    args = parser.parse_args()

    print(f"Running with: {args}")

    database = pd.read_csv(args.input_file)

    for i, v in database.iterrows():
        # print("\n")
        # print(v["input"])
        # print(v["input"])
        # zarr_convert(
        print(
            v["input"], 
            v["output"], 
            v["microscope"],
            float(v["max_size(MB)"]),
            int(v["scene_index"]),
            v["channel_names"],
            v["channel_colors"]
            )

if __name__ == "__main__":
    main()
