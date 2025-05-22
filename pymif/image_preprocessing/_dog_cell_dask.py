import numpy as np
import dask.array as da
from ._dog_cell import dog_cell
import pyclesperanto_prototype as cle

def dog_cell_dask(
        input_image, 
        chunk_size = (512,512,512),
        overlap = (32,32,32),
        cell_diameter = [10.,10.,10.],
        large_scale = np.sqrt(2)/(1.+np.sqrt(2)),
        small_scale = 1./(1.+np.sqrt(2))
        ):

    '''Difference of gaussian optimized to enhance structures of size "cell_diameter".
    Performs DoG blurs with sigma equal to "large_scale" and "small_scale" times "cell_diameter".

    Parameters
    ----------
    input_image : numpy array
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
    
    # print(input_image.shape)
    tiles = da.from_array(input_image, chunks=chunk_size)
    # print(tiles)

    tile_map = da.map_overlap(
                    dog_cell, 
                    tiles,
                    cell_diameter=cell_diameter,
                    large_scale=large_scale, 
                    small_scale=small_scale,
                    depth=overlap,
                    trim=True
                    )
    # print('////////////////////////////////////////')
    result = tile_map.compute()
    # print(result.shape)

    return result

if __name__ == '__main__':
    print('ciao')

    input_image = (256*np.random.rand(512,512,512)).astype(np.uint16)
    print(input_image.shape)
    result = dog_cell_dask(
        input_image, 
        chunk_size=(128,128,128), 
        overlap=(16,16,16), 
        cell_diameter=(2,2,2)
        )

    print(result)
    print(result.shape)
