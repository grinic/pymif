This folder contains some workflow examples on how to run the Opera PE scripts and functions.

1. Path to the pymif code should be updated using `sys.path.append()` to point to the directory on your machine where the pymif code has been saved

1. Parameters of the `compile_condition_...` function:
    * `path`: location of the experiment folder
    * `conditions`: only available for `compile_conditions` script. 8x12 list that contains the conditions for each well in the plate.
    * `channel_order`: order of fluorescence/brightfield as saved by PE. Should be a list of the form [1,0,2]. If using the `extract_channel_order`, the function will arrange the channels automatically starting from brightfield, and increasing fluorescence wavelength. E.g. [Bf, 488, 561, 640].
    * `luts_name`: name of the luts for ImageJ visualization.
    * `df`: metadata extracted from the csv file.
    * `downsample`: optional downsampling of the images to reduce disk space. E.g. 0.5 will reduce the image size by a factor of 2 in both directions.
    * `ff_mode`: can be "None" (no flat field correction), "slide" (manually provided flat field images), or "PE" (automatically extracted from the PE metadata files).
    * `image_folder`: name of the subdirectory where the .tif files are stored.
    * `outfolder`: name of the subdirectory where the compiled data will be saved.
    * `whoch_proj`: Z projection to be applied (if acquired a 3d stack). Can be "none" (save the whole 3D stack), "mip" (maximmum intensity projection), or "mean" (mean intensity projection). Only available for `compile_conditions`.

