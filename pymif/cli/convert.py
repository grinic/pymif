"""
Command-line interface for the pymif package to convert a single dataset
To run:
>> conda activate pymif
(pymif) >> ./convert_cli.py -i <input> -z <zarr> -m <microscope>
"""

import argparse
from argparse import RawTextHelpFormatter
import os
import re
import time
from matplotlib.colors import cnames
from .auto_zarr_convert import zarr_convert, parse_color

def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for the pymif package to convert a single dataset in zarr.",
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("--input_path", "-i", 
                        required=True, 
                        help="Path to the input file."
                        )
    parser.add_argument("--zarr_path", "-z", 
                        required=True, 
                        help="Path to the output zarr."
                        )
    parser.add_argument("--microscope", "-m", 
                        required=True, 
                        help="Microscope. One of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\"."
                        )
    
    parser.add_argument("--max_size", "-ms", 
                        required=False, default=100, 
                        help="Max chunk size in MB."
                        )
    parser.add_argument("--scene_index", "-si", 
                        required=False, default=-1, 
                        help="Scene index for .czi files."
                        )
    parser.add_argument("--channel_names", "-cn", 
                        required=False, default=None, 
                        nargs="+", help="Name of channels."
                        )
    parser.add_argument("--channel_colors", "-cc", 
                        required=False, 
                        type=parse_color, default=None,
                        nargs="+", help="Colors of channels (hex or matplotlib color name)."
                        )

    args = parser.parse_args()

    print(f"Running with: {args}")

    zarr_convert(
        args.input_path, 
        args.zarr_path, 
        args.microscope,
        float(args.max_size),
        int(args.scene_index),
        args.channel_names,
        args.channel_colors
        )


if __name__ == "__main__":
    main()
