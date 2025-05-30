import h5py
import numpy as np

def open_h5(fname, downscale=1):

    f = h5py.File(fname, 'r')
    dset = np.array(f['Data']).astype(np.uint16)

    dset = dset[:,::downscale,::downscale]

    return dset
