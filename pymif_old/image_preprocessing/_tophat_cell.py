import numpy as np
import pyclesperanto_prototype as cle

def tophat_cell(
        img, 
        cell_diameter = [5.,20.,20.],
        scale = 2.,
        ):

    top_hat_radius = cell_diameter * scale
    
    assert (top_hat_radius.ndim==1)&(len(top_hat_radius)==3), 'Detected incompatible top_hat_rad definition!'

    # print('Push to GPU...')
    img_gpu = cle.push(img)

    # correct for scattering and increased autofluorescence in some regions of the sample
    # print('TopHat...')
    img_tophat = cle.top_hat_box(img_gpu, 
                            radius_x=top_hat_radius[2], 
                            radius_y=top_hat_radius[1], 
                            radius_z=top_hat_radius[0])

    # print('Clear GPU memory')
    img_gpu = None

    # print('Done.')
    return cle.pull(img_tophat)