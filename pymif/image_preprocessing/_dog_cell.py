import numpy as np
import pyclesperanto_prototype as cle

# cle.set_wait_for_kernel_finish(True)

def dog_cell(
        img, 
        cell_diameter = [5.,20.,20.],
        large_scale = np.sqrt(2)/(1.+np.sqrt(2)),
        small_scale = 1./(1.+np.sqrt(2))
        ):

    '''Difference of gaussian optimized to enhance structures of size "cell_diameter".
    Performs DoG blurs with sigma equal to "large_scale" and "small_scale" times "cell_diameter".

    Parameters
    ----------
    img : numpy array
       Input image
    cell_diameter : optional, default [5,20,20]
       Approximater diameter of structures/cells to be enhanced
    large_scale : optional, default sqrt(2)/(1+sqrt(2))

    small_scale : optional, default 1/(1+sqrt(2))


    Returns
    -------
    img_DoG : numpy array

    '''

    cell_diameter = np.array(cell_diameter)
    
    assert (cell_diameter.ndim==1)&(len(cell_diameter)==3), 'Detected incompatible cell_diameter definition!'

   #  print('Push to GPU...', img.shape, cell_diameter.shape, large_scale)
    img_gpu = cle.push(img)

   #  print('Sigma1...', img_gpu.shape)
    d1 = cell_diameter*small_scale
    blurred1 = cle.gaussian_blur(img_gpu, sigma_x=d1[2], sigma_y=d1[1], sigma_z=d1[0])

   #  print('Sigma2...', blurred1.shape)
    d2 = cell_diameter*large_scale
    blurred2 = cle.gaussian_blur(img_gpu, sigma_x=d2[2], sigma_y=d2[1], sigma_z=d2[0])

   #  print('DoG...', blurred2.shape)
    img_DoG = cle.subtract_images(blurred1, blurred2)#cle.pull(blurred1)-cle.pull(blurred2)
    
    del img_gpu, blurred1, blurred2

   #  print('Done.', img_DoG.shape)
    return cle.pull(img_DoG)
