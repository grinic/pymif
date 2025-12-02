"""
Command-line interface for the pymif package to convert a single dataset
To run:
>>> conda activate pymif
(pymif) >>> ./convert_cli.py -i <input> -z <zarr> -m <microscope>
"""

import argparse
from argparse import RawTextHelpFormatter
from .auto_zarr_convert import zarr_convert, parse_color

def main():
    """Command-line interface for the pymif package to convert a single dataset in zarr.

    More info with:

    >>> pymif-2zarr --help
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the pymif package to convert a single dataset to zarr.",
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
                        help="Microscope.\nOne of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"zarrv04\", \"zarr\"."
                        )
    
    parser.add_argument("--max_size", "-ms", 
                        required=False, default=100, 
                        help="[Optional] Max chunk size in MB. \nDefault: 100"
                        )
    parser.add_argument("--scene_index", "-si", 
                        required=False, default=-1, 
                        help="[Optional] Scene index for .czi files. \nDefault: -1"
                        )
    parser.add_argument("--channel_names", "-cn", 
                        required=False, default=None, 
                        nargs="+", help="[Optional] Name of channels.\nExample: \"-cn bf gfp rfp\"\nDefault: None"
                        )
    parser.add_argument("--channel_colors", "-cc", 
                        required=False, 
                        type=parse_color, default=None,
                        nargs="+", help="[Optional] Colors of channels (hex or matplotlib color name)\nExample: \"-cc 0000FF cyan 00ff00\")\nDefault: None"
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
