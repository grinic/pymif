import pyclesperanto_prototype as cle
import numpy as np

def intensity_corr_cell(
    input_image, 
    cell_diameter = [5.,20.,20.],
    scale = 10.
    ):

    '''
    Divide the image by its gaussian blur version with a sigma equal to "scale" times "cell_diameter".
    '''

    cell_diameter = np.array(cell_diameter)
    
    input_gpu = cle.push(input_image)

    # perform intensity correction
    blurred_gpu = cle.gaussian_blur(
        input_gpu, 
        sigma_x=cell_diameter[2]*scale, 
        sigma_y=cell_diameter[1]*scale, 
        sigma_z=cell_diameter[0]*scale
        )
    blurred_gpu = blurred_gpu/float(np.mean(blurred_gpu))
    output_gpu = cle.divide_images(input_gpu, blurred_gpu)

    # output_image = cle.pull(output_gpu)
    # blurred = cle.pull(blurred_gpu)

    del input_gpu, blurred_gpu

    return cle.pull(output_gpu)
