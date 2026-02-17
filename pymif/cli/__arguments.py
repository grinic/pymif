__all__ = ['_parse_arguments']
import argparse
import os
import textwrap
import re
from matplotlib.colors import cnames

class MultilineDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass

HEX_PATTERN = re.compile(r'^#?[0-9a-fA-F]{6}$')

# Valid type of's
def valid_input_path(x):
    if x is not None:
        if os.path.isdir(x) or os.path.isfile(x):
            return os.path.abspath(x)
        else:
            raise argparse.ArgumentTypeError(f'Input path {x} is not a valid directory')
    else: 
        return None

def valid_output_path(x):
    print(x)
    print((not os.path.isdir(x)) and (not os.path.isfile(x)))
    if x is not None:
        print((not os.path.isdir(x)) and (not os.path.isfile(x)))
        if (not os.path.isdir(x)) and (not os.path.isfile(x)):
            return os.path.abspath(x)
        else:
            raise argparse.ArgumentTypeError(f'Output path {x} is not a valid directory')
    else: 
        return None

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

def _parse_arguments():

     
    parser = argparse.ArgumentParser(
        description= """\
            Welcome fellow MIF users!
        """,
        formatter_class= argparse.RawDescriptionHelpFormatter
    )

    # Sub-parsers
    subparsers = parser.add_subparsers(
        title= 'Runmodes',
        description= """\
            PyMIF has TWO main runmodes, each with different and specific arguments.
            Please consult each runmode's help manual before running any of them.
            Enjoy PyMIF!
        """,
        required= True,
    )

    #####################################################################################
    # Single convert 2 zarr parser
    single_convert_parser = subparsers.add_parser(
        '2zarr',
        help= 'Convert to zarr format a single image.',
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )
    single_convert_parser.add_argument(
        '--runmode',
        help= argparse.SUPPRESS,
        default= 0,
        type= int
    )

        # Optional args
    single_convert_parser.add_argument(
        '-ms', '--max_size',
        required= False,
        default= 100,
        help= 'Max chunk size in MB.',
        type= float
    )
    single_convert_parser.add_argument(
        '-si', '--scene_index',
        required= False,
        default= 0,
        help= 'Scene index for .czi files.',
        type= int
    )
    single_convert_parser.add_argument(
        '-cn', '--channel_names',
        required= False,
        default= None,
        nargs= '+',
        help= 'Name of channels. Example: -cn bf gfp rfp'
    )
    single_convert_parser.add_argument(
        '-cc', '--channel_colors',
        required= False,
        default= None,
        type= parse_color,
        nargs= '+',
        help= 'Color(s) of channel(s). It needs to be hex or matplotlib color name. Example: -cc 0000FF cyan 00ff00'
    )

        # Required args
    requiredNamed = single_convert_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument(
        '-i', '--input_path',
        required= True,
        help= 'Path to input file.',
        type= valid_input_path
    )
    requiredNamed.add_argument(
        '-z', '--zarr_path',
        required= True,
        help= 'Path to output zarr.',
        type= valid_output_path
    )
    requiredNamed.add_argument(
        '-m', '--microscope',
        required= True,
        help= 'Microscope used in previous analysis.',
        choices= ['luxendo', 'opera', 'viventis', 'opera', 'zeiss', 'zarrv04', 'zarr', 'scape'],
        type= str
    )

    #####################################################################################
    # Batch convert 2 zarr parser
    long_block = """\
        Convert to zarr format a batch of images.
        The INPUT_FILE is a .csv file of the form:
        input              | microscope  | output           | max_size(MB) | scene_index | channel_colors | channel_names
        /path/to/input_1   | opera       | /path/to/zarr_1  | 100          | 0           | lime white     | gfp bf
        /path/to/input_2   | viventis    | /path/to/zarr_2  | 100          |             | 000FF FF00FF   |
        ...
        /path/to/input_n   | viventis    | /path/to/zarr_n  | 100          | 0           |                |
        All column headers are mandatory, but values can be empty
        channel_colors can be hex code or valid matplotlib colors.
        An example .csv file can be found in /pymif/examples folder.
    """
    batch_convert_parser = subparsers.add_parser(
        'batch2zarr',
        help= 'Convert a batch of images to zarr (short one-line help). A dedicated instructions file needs to be generated beforehand. Proceed to the batch2zarr help manual for the example use.',
        description= textwrap.dedent(long_block),
        formatter_class= MultilineDefaultsHelpFormatter
    )
    batch_convert_parser.add_argument(
        '--runmode',
        help= argparse.SUPPRESS,
        default= 1,
        type= int
    )
        # Required args
    requiredNamed = batch_convert_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument(
        '-i', '--input_file',
        required= True,
        help= 'Path to input file.',
        type= valid_input_path
    )

    #####################################################################################
    # Possible other runmodes

    #####################################################################################
    args = parser.parse_args()
    return args