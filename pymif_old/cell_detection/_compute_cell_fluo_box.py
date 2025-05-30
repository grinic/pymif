import numpy as np
import tqdm

def compute_cell_fluo_box(df, imgs, cell_diameter=[2.,10.,10.], ch_names=None):
    """
    
    Parameters
    ----------
    df : dataframe
        pandas dataframe containing location info.
    imgs : list, str
        list of images.
    ch_names : list, optional
        name of channels. Must match number and order of files given. The default is None.
        
    Returns
    -------
    df : dataframe
        return dataframe populated with fluorescence intensity values.

    """
    ### load fluo images
    # imgs = np.stack([imread(f) for f in tqdm.tqdm(files)])
    
    n_cells = len(df)
    n_ch = len(imgs)

    df1 = df.copy()
    
    if ch_names == None:
        ch_names=['ch_'+str(i) for i in range(len(n_ch))]
    else:
        ch_names = [str(i) for i in ch_names]
        
    assert len(ch_names)==n_ch

    # initialize with empty fluorescence intensities
    for ch_name in ch_names:
        df1[ch_name] = 0.

    ### compute fluo intensity
    for i in tqdm.tqdm(range(n_cells), total=n_cells):
        cell = df1.loc[i]
        x,y,z = int(cell.x), int(cell.y), int(cell.z)
        r_z = int(np.round(cell_diameter[0]/2.))
        r_y = int(np.round(cell_diameter[1]/2.))
        r_x = int(np.round(cell_diameter[2]/2.))
        
        fluos = [np.mean(img[z-r_z:z+r_z,
                             y-r_y:y+r_y,
                             x-r_x:x+r_x].astype(float)) for img in imgs]
        
        for ch_name, fluo in zip(ch_names, fluos):
            df1.loc[i,ch_name] = fluo
            
    return df1
