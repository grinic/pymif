import numpy as np
import dask.array as da
from ._tophat_cell import tophat_cell

def tophat_cell_dask(
                    input_image, 
                    chunk_size = (512,512,512),
                    overlap = (32,32,32),
                    cell_diameter = [10.,10.,10.],
                    scale = 2.,
                    ):
    '''Function to rescale a inostropic image to isotropic resolution using dask for tile (chunk) processing.

    Parameters
    ----------
    input_image: 3D numpy array
            dimensions should be arranged as ZYX
    chunk_size: iterable, int, optional, default: (128,512,512)
            chunk dimensions
    overlap: iterable, int, optional, default: (8,32,32)
            overlap between tiles (avoid edge effects)
    cell_diameter: iterable, float, default: [10., 10., 10.]
            estimated diameter of the cells in every dimension.
    scale: float, default: 2.
            Box size on which to make tophat in units of cell diameter.

    Returns
    -------
    result: 3D numpy array
            the tophat image
    
    '''

    cell_diameter = np.array(cell_diameter)
    
#     print(input_image.shape)
    tiles = da.from_array(input_image, chunks=chunk_size)

    tile_map = da.map_overlap(
                    tophat_cell, 
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
    result = tophat_cell_dask(
        input_image, 
        chunk_size=(128,128,128), 
        overlap=(16,16,16), 
        cell_diameter=(8,8,8)
        )

    print(result)
    print(result.shape)
    
