import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_li as threshold
from skimage.registration import phase_cross_correlation

def correct_chromatic_shift(images, ref_channel=0):
    n_ch = len(images)

    images_shifted = [None for i in range(n_ch)]
    shifts = [None for i in range(n_ch)]

    for i in range(n_ch):
        shift, error, phasediff = phase_cross_correlation(images[ref_channel], images[i])
        shifts[i] = int(-shift[0]) # keep just the shift in the Z dimension

    # print(shifts)
    min_shift = np.min(shifts)
    shifts = [s-min_shift for s in shifts]
    # print(shifts)

    # correct images by removing the appropriate number of planes
    images_shifted = [images[i][shifts[i]:-(np.max(shifts)-shifts[i]+1),:,:] for i in range(n_ch)]

    return images_shifted, shifts
