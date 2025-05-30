import os, time, sys
import numpy as np
import pandas as pd
sys.path.append('/g/mif/people/gritti/code/pymif')
from pymif_old import luxendo
from pymif_old import io
# from measure_somitoid_volume import compute_full_volume
# from measure_ch_volume import compute_ch_volume, compute_tbx6_volume

if __name__ == "__main__":
    # '''
    # To run this script:

    # >>> python run_cell_analysis.py --share /mif-users --radius 1,1,1 --min_distance 1
    # '''

    # def list_of_floats(arg):
    #     l = [float(f) for f in arg.split(",")]
    #     # print("---",l)
    #     return l

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--share", 
    #                     type=str, default="/g/ebisuya", 
    #                     help="Server share. /g/ebisuya (Linux) or //ebisuya.embl.es/ebisuya (Windows)")
    # parser.add_argument("--input_folder",
    #                     type=str, default="input-tray/Cell_count/data/primary")
    # parser.add_argument("--output_folder",
    #                     type=str, default="input-tray/Cell_count/data/processed_new")
    # parser.add_argument("--exp_folder",
    #                    type=str, default="MC20230628_sampleA_650_DAPI-405_SOX2-488_TBX6-594_BRA-647/2023-09-28_165114")

    # parser.add_argument("--desired_scale", 
    #                     type=list_of_floats, default=[4., 4., 4.],
    #                     help="Desired pixel size in um after downsampling (provide a value for each ZYX dimension).")

    # args = parser.parse_args()

    # share = args.share
    # input_folder = args.input_folder
    # output_folder = args.output_folder
    # exp_folder = args.exp_folder
    # desired_scale = args.desired_scale

    ###############################################################

    ### TO SET exp folder manual:
    share = "/g/mif"
    input_folder = "people/gritti/code/test_data/luxendo/2023-06-28_165114"
    output_folder = "people/gritti/code/test_data/luxendo/2023-06-28_165114/output"
    desired_scale = 4. # um/pxl in ZYX, always isotropic!

    ###############################################################

    path = os.path.join(share, input_folder)
    exp_name = os.path.split(path)[-1]

    output_path = os.path.join(share, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ### extract metadata and save
    meta = luxendo.extract_meta(path)
    print(meta)

    ### read, downsample and resave h5 files as tif
    start = time.time()

    images = [None for i in range(meta["n_ch"])]

    for i in range(meta["n_ch"]):
        image = luxendo.open_downsample_h5(path, ch=i, sample_meta=meta, desired_scale=desired_scale)
        _ = io.save_ij_tiff(
                    os.path.join(output_path, "ch-%d_%s.tif"%(i, meta["ch-%d_name"%i])),
                    image,
                    ax_names = "ZYX",
                    ax_scales = 3*[desired_scale],
                    ax_units = 3*["um"],
            )
        images[i] = image

    print("Time to read images: %.2f minutes."%((time.time()-start)/60))

    ### correct chromatic shift
    start = time.time()

    images_corr, shifts = luxendo.correct_chromatic_shift(images)

    print("Time to shift images: %.2f minutes."%((time.time()-start)/60))

    ### save metadata as csv and corrected images as tif
    start = time.time()

    meta["ch_shift_unit"] = "planes"

    for i in range(meta["n_ch"]):
        meta["ch-%d_shift"%i] = shifts[i]

        _ = io.save_ij_tiff(
                    os.path.join(output_path, "ch-%d_%s_shifted.tif"%(i, meta["ch-%d_name"%i])),
                    images_corr[i],
                    ax_names = "ZYX",
                    ax_scales = 3*[desired_scale],
                    ax_units = 3*["um"],
            )

    meta = pd.Series(meta)
    meta.to_csv(os.path.join(output_path, "meta.csv"), header=False)

    print("Time to save shifted images: %.2f minutes."%((time.time()-start)/60))





