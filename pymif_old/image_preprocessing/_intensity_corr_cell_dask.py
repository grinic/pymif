import numpy as np
import dask.array as da
from ._intensity_corr_cell import intensity_corr_cell

def intensity_corr_cell_dask(
    input_image, 
    chunk_size = (512,512,512),
    overlap = (32,32,32),
    cell_diameter = [10.,10.,10.],
    scale = 10.
    ):

    '''
    Divide the image by its gaussian blur version with a sigma equal to "scale" times "cell_diameter".
    '''
    
    cell_diameter = np.array(cell_diameter)
    
    # print(input_image.shape)
    tiles = da.from_array(input_image, chunks=chunk_size)

    tile_map = da.map_overlap(
                    intensity_corr_cell,
                    tiles,
                    cell_diameter=cell_diameter,
                    scale=scale, 
                    depth=overlap)
    result = tile_map.compute()

    return result


if __name__ == '__main__':
    print('ciao')

    input_image = 256*np.random.rand(512,512,512)
    print(input_image.shape)
    result = intensity_corr_cell_dask(
        input_image, 
        chunk_size=(128,128,128), 
        overlap=(16,16,16), 
        cell_diameter=[8,8,8]
        )

    print(result)
    print(result.shape)
