import pyclesperanto_prototype as cle

def isores(input_image, pixel_size, iso_res):
    '''
    Function to rescale a anisotropic image to isotropic resolution.

    Parameters:
    -----------
    input_image: ZYX
    pixel_size: ZYX in um/pxl
    iso_res: desired isotropic resolution in um/pxl

    Returns:
    --------
    resampled: image with isotropic resolution
    
    '''
    print(input_image.shape)
    input_gpu = cle.push(input_image)

    resampled_gpu = cle.scale(input_gpu, resampled_gpu, 
              factor_x=pixel_size[2]/iso_res, 
              factor_y=pixel_size[1]/iso_res, 
              factor_z=pixel_size[0]/iso_res, 
              centered=False, linear_interpolation=True)

    # show(resampled_gpu)
    print(resampled_gpu.shape)

    resampled = cle.pull(resampled_gpu)

    del input_gpu, resampled_gpu

    return resampled
    
