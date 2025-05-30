import numpy as np
from skimage.io import imread, imsave
import glob, os, tqdm
from imagej_fun import imagej_metadata_tags, make_lut
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_uint
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################

def binary2volume(
                path, pxl_size=[2.,0.76,0.76] 
            ):

    voxel_volume = 1e-9*np.prod(pxl_size) # in mm^3

    flist = glob.glob(os.path.join(path,'mask_channel*.tif'))
    flist = [f for f in flist if 'MIP' not in f]
    flist.sort()
    N = len(flist)

    file_name = os.path.basename(flist[0])
    file_path = os.path.dirname(flist[0])

    # compute volume of gastruloid
    org_mask = imread(os.path.join(file_path,'mask_total.tif'))
    org_mask = org_mask.astype(float)/np.max(org_mask)
    V_gastr = float(np.sum(org_mask)) * voxel_volume
    # print('Total volume:',V_gastr)

    volumes = np.array([0. for i in flist])
    overlap = np.zeros((N,N))

    # compute volume of every channel
    masks = [0 for i in flist]
    i = 0
    for filename in tqdm.tqdm(flist):
        file_name = os.path.basename(filename)
        file_path = os.path.dirname(filename)
        file_root, file_ext = os.path.splitext(file_name)

        masks[i] = imread(os.path.join(file_path,file_name))
        masks[i] = masks[i].astype(float)/np.max(masks[i])
        v = float(np.sum(masks[i])) * voxel_volume
        # print('Volume ch%d'%i,v)

        # print(v/V_gastr)
        volumes[i] = v
        # print('Volume fraction ch%d'%i,v/V_gastr)
        i+=1

    # compute overlap between pairs of channels
    for i in range(N):
        for j in range(N):
            if j>i:
                m = masks[i] * masks[j]
                v = float(np.sum(m))
                overlap[i,j] = v * voxel_volume

    # save in dataframe
    vals = [path,V_gastr]
    names = ['name','V_tot']
    for i, v in enumerate(volumes):
        vals.append(v)
        names.append('V_ch%d'%i)
    for i in range(N):
        for j in range(N):
            if j>i:
                vals.append(overlap[i,j])
                names.append('V_ch%d-%d'%(i,j))
    df = pd.DataFrame([vals], columns=names)
    # print(df)

    return df

################################################################################

if __name__=='__main__':

    paths = [ 
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_114830'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_115427'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_120041'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_121905'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_122616'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_123405'),
            ]

    for path in tqdm.tqdm(paths):
        binary2volume(path)


