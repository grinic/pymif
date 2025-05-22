import numpy as np
from skimage.io import imread, imsave
import tifffile
from skimage.transform import rescale
import  os, string, tqdm
import pandas as pd
from ...imagej_funs._make_lut import make_lut
from ...imagej_funs._imagej_metadata_tags import imagej_metadata_tags
from ..pe_io._extract_ffc_info import extract_ffc_info

def compile_conditions(
        path, 
        conditions, 
        channel_order, 
        luts_name,
        df,
        ff_mode = 'PE', 
        ffs=None,
        downsample=1.,
        image_folder = os.path.join("Images"),
        outfolder = 'compiled',
        which_proj = 'none',
        ):

    '''This function combines images of a 96WP acquired by PE.

    Parameters
    ----------
    path: string
            a string containing the path t the experiment folder. Has to point to the "Images" folder
    conditions: iterable, string
            list of conditions for every well (shape 12x8)
    channels_order: iterable, int
            the order in which the channels have to be arranged in the output images.
            E.g. if PE acquires GFP first and BF next, and you want BF-GFP, then the order is [1,0]
    luts_name: iterable, string
            list of colors to be used to show the channels

    Returns
    -------
    a "compiled" folder, containing the conditions subfolders, containing 1 multichannel tif file for every well in the experiment

    NOTE:
    This script assume the experiment contains just one FOV per well!
    '''
    # ff_mode: 'PE' for PE FF correction, use 'slide' for autofluorescence slide, use 'None' for no correction
    if ff_mode == "slide":
        ffs = [ffs[i]/np.median(ffs[i]) if ffs[i] is not None else 1. for i in range(len(channel_order))]
    elif ff_mode == "PE":
        ffs_info = extract_ffc_info(path, channel_order)
        ffs = [ff_info["ff_profile"] for ff_info in ffs_info]
    else:
        ffs = [1. for i in channel_order]

    # find out all wells
    wells = df.groupby(['row','col']).size().reset_index()    

    # define well id to convert e.g. r01c01 into A01
    d = dict(enumerate(string.ascii_uppercase, 1))


    pbar = tqdm.tqdm(wells.iterrows())
    conversion = pd.DataFrame({})
    for p in pbar:
        # print(p)
        r = int(p[1].row)
        c = int(p[1].col)
        well = d[r]+'%02d'%c
        
        cond = conditions[int(p[1].col)-1][int(p[1].row)-1]
        
        pbar.set_description(well + ' ' + cond)
        pbar.update()
        
        outpath = os.path.join(path,outfolder,cond)
        if not os.path.exists(outpath):
            os.makedirs(outpath)    

        df_well = df[(df.row==r)&(df.col==c)]
        # print(df_well)
        
        if len(df_well)>0:
            
            stack = []
            for k, ch in enumerate(channel_order):
                df_pos_ch = df_well[df_well.channel==(ch+1)]
                df_pos_ch = df_pos_ch.sort_values(by='Zpos')
                
                # print('-'*25,'ch:',ch)
                # print(df_pos_ch)
                
                stack_ch = np.stack([rescale(imread(os.path.join(path,image_folder,img_file))/ffs[k], [downsample, downsample], order=1, preserve_range=True, anti_aliasing=True) for img_file in df_pos_ch.filename])

                stack.append(stack_ch)

            # order channels
            stacks = np.stack(stack).astype(np.uint16)

            if which_proj=='mip':
                tosave = []
                for k, s in enumerate(stacks):
                    if k==0:
                        tosave.append(np.min(s, 0))
                    else:
                        tosave.append(np.max(s, 0))
                tosave = np.stack(tosave).astype(np.uint16)
            elif which_proj=='mean':
                tosave = []
                for k, s in enumerate(stacks):
                    if k==0:
                        tosave.append(np.min(s, 0))
                    else:
                        tosave.append(np.mean(s, 0))
                tosave = np.stack(tosave).astype(np.uint16)
            else: 
                tosave = np.swapaxes(stacks, 0, 1).astype(np.uint16)

            # tosave = np.moveaxis(tosave,0,-1)
            tosave = tosave.astype(np.uint16)

            # create imagej metadata with LUTs
            luts_dict = make_lut(luts_name)
            # luts_dict = make_lut_old()
            ijtags = imagej_metadata_tags({'LUTs': [luts_dict[lut_name] for lut_name in luts_name]}, '>')

            outname = well+'.tif'

            raw = pd.DataFrame({'filename':[outname],
                                'row_idx':[r],
                                'col_idx':[c],
                                'condition':cond,
                                'pixelSize':df_well.pixelSize.values[0],
                                })
            for k, ch in enumerate(channel_order):
                raw["ch%d_name"%k] = df_well.chName.values[ch]
                raw["ch%d_wavelength"%k] = df_well.chWavelength.values[ch]

            conversion = pd.concat([conversion,raw], ignore_index=True)

            tifffile.imwrite(os.path.join(outpath,outname), tosave, byteorder='>', imagej=True,
                            resolution=(1./df_well.pixelSize.values[0],1./df_well.pixelSize.values[0]),
                            metadata={
                                'mode': 'composite', 
                                'unit': 'um',
                                }, 
                            extratags=ijtags)

    conversion.to_csv(os.path.join(path,outfolder, 'metadata.csv'))
