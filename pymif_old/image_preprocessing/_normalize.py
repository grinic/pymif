import numpy as np
from skimage.util import img_as_float32

def normalize(
            img, 
            contrast_mode = 'percs',
            percentiles = [3,99.99], 
            percentiles_downsample = [2,2,2],
            abs_lims = [0,2**16-1],
            ):

    percentiles = np.array(percentiles)
    percentiles_downsample = np.array(percentiles_downsample)
    
    assert (percentiles.ndim==1)&(len(percentiles)==2), 'Detected incompatible percentiles definition!'
    assert (percentiles_downsample.ndim==1)&(len(percentiles_downsample)==3), 'Detected incompatible percentiles_downsample definition!'
    
    
    # to make it more robust, normalise values before finding peaks.
    # print('To float...')
    img_float = img_as_float32(img)

    # find 3 and 97 percentiles (typically used by image analysis, otherwise try 10 and 90 percent)
    # print('Percs...')
    img_down = img_float[::percentiles_downsample[0], ::percentiles_downsample[1], ::percentiles_downsample[2]]
    try:
        if contrast_mode == 'percs':
            perc = percentiles
            lims_val = np.percentile(img, tuple(perc))
        elif contrast_mode == 'absolute':
            lims_val = abs_lims
    except ValueError:
        print("Contrast mode not valid...")

    # print('Normalize...')
    img_float = (img_float - lims_val[0]) / (lims_val[1] - lims_val[0])

    # print('Clip...')
    img_float = np.clip(img_float, 0, 1)
    
    # print('Done.')
    return img_float