import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle

def cell_local_max(
            input_image, 
            cell_diameter,
            thr_DoG=0.01, 
            gpu_downsample=[1,1,1],
            lims=np.array([[50,50],[100,100],[100,100]])
            ):

    gpu_downsample = np.array(gpu_downsample)
    cell_diameter = np.array(cell_diameter)

    assert (gpu_downsample.ndim==1)&(len(gpu_downsample)==3), 'Detected incompatible gpu_downsample definition!'
    assert (cell_diameter.ndim==1)&(len(cell_diameter)==3), 'Detected incompatible cell_diamater definition!'

    print('Detect local maxima...')
    detected_spots = cle.detect_maxima_box(input_image[::gpu_downsample[0],::gpu_downsample[1],::gpu_downsample[2]], 
                        radius_x = np.sqrt(2)*cell_diameter[2], 
                        radius_y = np.sqrt(2)*cell_diameter[1], 
                        radius_z = np.sqrt(2)*cell_diameter[0])
    selected_spots = cle.binary_and(input_image>thr_DoG, detected_spots)
    
    print('Detect coordinates...')
    p = np.where(selected_spots)

    df = pd.DataFrame({
                        'z': p[0].astype(float)*gpu_downsample[0],
                        'y': p[1].astype(float)*gpu_downsample[1],
                        'x': p[2].astype(float)*gpu_downsample[2],
                        })

    print('Filter coordinates...')
    df = df[(df.z>lims[0,0])&(df.z<(input_image.shape[0]*gpu_downsample[0]-lims[0,1]))&
            (df.y>lims[1,0])&(df.y<(input_image.shape[1]*gpu_downsample[1]-lims[1,1]))&
            (df.x>lims[2,0])&(df.x<(input_image.shape[2]*gpu_downsample[2]-lims[2,1]))]

    del detected_spots, selected_spots

    print('Done.')
    return df