from pymif.cli.__arguments import _parse_arguments
from pymif.cli.auto_zarr_convert import zarr_convert, parse_color
import pandas as pd

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