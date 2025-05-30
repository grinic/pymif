import os, glob
import h5py
import numpy as np
from skimage.transform import rescale

def open_downsample_h5(path, ch, sample_meta, desired_scale=1):

    print("\tDesired isotropic scale in micrometers (ZYX):", desired_scale)

    # print(proc_folder)

    if os.path.exists(path+"/processed"):
        file_folder = [ f.path for f in os.scandir(path+"/processed/") if f.is_dir() ][0]
        flist = glob.glob(os.path.join(file_folder, "*.lux.h5"))
        flist.sort()
        file_name = flist[ch]
        print("\tLoading from this file:",file_name)

        ### find out the downsampling in each dataset
        with h5py.File(file_name, "r") as f:
            keys = [k for k in f.keys() if "Data" in k]
        dataset_names = ["" for i in keys]
        for i in range(len(keys)):
            if keys[i]=="Data":
                dataset_names[i] = keys[i]+"_1_1_1"
            else:
                dataset_names[i] = keys[i]
        # print(dataset_names)
        dims = [np.array(k.split("_")[1:]).astype(int) for k in dataset_names]

        ### extract the downsampling in XY
        downsampling = [d[0] for d in dims]
        # print(downsampling)

        ### compute the desired downsampling from the desired scaling
        downXY_desired = desired_scale/sample_meta["scale_1_1_1_y"]

        ### find which available dataset has just a little bit more resolution than the desired
        downXY_chosen = 0
        for d in downsampling:
            if d < downXY_desired and d > downXY_chosen:
                downXY_chosen = d
        idx_downsampling = downsampling.index(downXY_chosen)
        data_key = dataset_names[idx_downsampling]

        ### find what is tha scale of that dataset in micrometers
        data_downsample = dims[idx_downsampling]
        scales = [sample_meta["scale_1_1_1_z"]*data_downsample[2], 
                  sample_meta["scale_1_1_1_y"]*data_downsample[1], 
                  sample_meta["scale_1_1_1_x"]*data_downsample[0]
                  ]

    else:
        print("\tNo processed folder found!")
        files_folder = [ f.path for f in os.scandir(path+"/raw/") if f.is_dir() ]
        assert len(files_folder)==sample_meta["n_ch"], "Expected %d channel, found %d folders!"%(sample_meta["n_ch"], len(files_folder))
        file_name = glob.glob(os.path.join(files_folder[ch], "*.lux.h5"))[0]
        print("\tLoading from this file:",file_name)

        ### in this case, you have to load the full res dataset, which is the only available.
        data_key = "Data"
        scales = [sample_meta["scale_1_1_1_z"], sample_meta["scale_1_1_1_y"], sample_meta["scale_1_1_1_x"]]


    with h5py.File(file_name, "r") as f:
        # print(f.keys())
        # print(dims)

        print("\tCh: %d -> Extracting \"%s\" dataset with scale (ZYX): %s micrometers"%(ch, data_key, scales))
        # print("\t", scales)
        # print("\t", desired_scale)
        dataset = f[data_key][()]
        downsample = [
            scales[0]/desired_scale, 
            scales[1]/desired_scale, 
            scales[2]/desired_scale
            ]
        print("\tUp/Downsampling (Rescaling) by a factor of (%.3f,%.3f,%.3f)"%(downsample[0], downsample[1], downsample[2]))
        dataset = rescale(dataset, 
                          downsample, 
                          order=1, 
                          preserve_range=True, 
                          anti_aliasing=True
                          )
        # dataset = np.stack([rescale(d, (1/downXY, 1/downXY), 
        #                   order=1, preserve_range=True, anti_aliasing=True) for d in tqdm.tqdm(dataset, total=dataset.shape[0])])
        dataset = dataset.astype(np.uint16)

    return dataset


def test_open_and_downsample_h5():
    path = os.path.join("/mif-users","Users","Laura_Bianchi",
                        "2023-11-20_170158","processed","20231204-072053_Task_1_sample0_control_C")
    meta = extract_meta(path)

    dataset = read_channel(path, ch=1, tp=12, sample_meta=meta, downXY=1)
