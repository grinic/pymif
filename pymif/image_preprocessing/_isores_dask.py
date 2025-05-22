import pyclesperanto_prototype as cle
import numpy as np
import dask.array as da

def isores_dask(
                    input_image, 
                    pixel_size, 
                    iso_res,
                    chunk_size = (128,512,512),
                    overlap = (8,32,32),
                    linear_interpolation = True
                    ):
    '''Function to rescale a inostropic image to isotropic resolution using dask for tile (chunk) processing.

    Parameters
    ----------
    input_image: 3D numpy array
            dimensions should be arranged as ZYX
    pixel_size: iterable, float
            dimensions should match input_image, ZYX in um/pxl
    iso_res: float
            desired isotropic resolution in um/pxl
    chunk_size: iterable, int, optional, default: (128,512,512)
            chunk dimensions
    overlap: iterable, int, optional, default: (8,32,32)
            overlap between tiles (avoid edge effects)
    linear_interpolation: bool, default: True
            whether to interpolate linearly when upscaling

    Returns
    -------
    result: 3D numpy array
            the reshaped image to isotropic resolution
    
    '''

    def rescale(chunk):
    
        # pprint(block_info)
        # push chunk to gpu
        chunk_gpu = cle.push(chunk)
        
        # rescaled chunk
        scaled = np.zeros((100,100,100))
        if chunk_gpu.shape[0] > 1:
            scaled_gpu = cle.scale(
                    chunk_gpu, 
                    factor_x=pixel_size[2]/iso_res, 
                    factor_y=pixel_size[1]/iso_res, 
                    factor_z=pixel_size[0]/iso_res, 
                    auto_size=True,
            #           centered=False, 
                    linear_interpolation=linear_interpolation
            )
            # pull chunk into memory
            scaled = cle.pull(scaled_gpu)
            del scaled_gpu
       
        # scaled_shape = scaled.shape
        
        # compute overlap in new iso_res dimension for manual trimming
        new_overlaps = (np.array(overlap)*np.array(pixel_size)/iso_res).astype(int)
        scaled = scaled[
                    new_overlaps[0]:-new_overlaps[0],
                    new_overlaps[1]:-new_overlaps[1],
                    new_overlaps[2]:-new_overlaps[2],
                    ]
        
        # print out some stuff
        # print(chunk.shape, scaled_shape,  new_overlaps, scaled.shape, np.array(pixel_size)/iso_res)
        # print("="*25)
        
        # free gpu memory
        del chunk_gpu
        
        return scaled

    print(input_image.shape)
    tiles = da.from_array(input_image, chunks=chunk_size)

    tile_map = da.map_overlap(rescale, tiles, depth=overlap, trim=False)
    result = tile_map.compute()

    return result
    
