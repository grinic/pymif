# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:23:19 2022

@author: gritti
"""

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale
import os, string, tqdm
import pandas as pd
from ...imagej_funs._make_lut import make_lut
from ...imagej_funs._imagej_metadata_tags import imagej_metadata_tags
from ..pe_io._extract_ffc_info import extract_ffc_info

def compile_conditions_multifields_timelapse(
        path, 
        channel_order, 
        luts_name, 
        df,
        ff_mode = 'PE',
        ffs=None,
        downsample=1.,
        image_folder = os.path.join("Images"), 
        outfolder = 'compiled',
        ):    
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
    for i, p in pbar:
        r = int(p.row)
        c = int(p.col)
        well = d[r]+'%02d'%c
        
        conversion = pd.DataFrame({})

        pbar.set_description(well)
        pbar.update()
        
        outpath = os.path.join(path, outfolder, well)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
            
        df_well = df[(df.row==r)&(df.col==c)]
        
        timepoints = list(set(df_well.timepoint))
        timepoints.sort()
        
        for timepoint in timepoints:
            df_tp = df_well[df_well.timepoint==timepoint]
        
            # find all fields inside this well
            fields = df_tp.groupby(['Ypos','Xpos']).size().reset_index()
            fields = fields.sort_values(by=['Ypos','Xpos'])
            l = list(set(fields.Ypos))
            l.sort()
            fields['Yidx'] = [l.index(v) for v in fields.Ypos]
            l = list(set(fields.Xpos))
            l.sort()
            fields['Xidx'] = [l.index(v) for v in fields.Xpos]
            
            for j, f in tqdm.tqdm(fields.iterrows(), total = len(fields)):
                x = f.Xpos
                y = f.Ypos
                xidx = f.Xidx
                yidx = f.Yidx
                
                df_pos = df_tp[(df_tp.Xpos==x)&(df_tp.Ypos==y)]
                
                # print('-'*50)
                # print(df_pos)
                
                if len(df_pos)>0:
    
                    # print('Images foudn')
                    stack = []
                    for k, ch in enumerate(channel_order):
                        df_pos_ch = df_pos[df_pos.channel==(ch+1)]
                        df_pos_ch = df_pos_ch.sort_values(by='Zpos')
                        print('-'*25,'x:',xidx,'y:',yidx,'ch:',ch)
                        print(df_pos_ch)
                        # [print(img_file) for img_file in df_pos_ch.filename]
                        # print([os.path.join(folder_raw,exp_folder,'Images',img_file) for img_file in df_pos_ch.filename])
                        stack_ch = np.stack([rescale(imread(os.path.join(path,image_folder,img_file))/ffs[k], [downsample, downsample], order=1, preserve_range=True, anti_aliasing=True) for img_file in df_pos_ch.filename])
                        stack.append(stack_ch.astype(np.uint16))
        
                    # order channels
                    stacks = np.array(stack).astype(np.uint16)
                    stacks = np.swapaxes(stacks, 0, 1)
        
                    # create imagej metadata with LUTs
                    luts_dict = make_lut(luts_name)
                    # luts_dict = make_lut_old()
                    ijtags = imagej_metadata_tags({'LUTs': [luts_dict[lut_name] for lut_name in luts_name]}, '>')
                    
                    outname = 'field%03d_tp%05d.tif'%(j,timepoint)
        
                    raw = pd.DataFrame({'tile_idx':[j],
                                        'filename':[outname],
                                        'row_idx':[yidx],
                                        'col_idx':[xidx],
                                        'timepoint':[timepoint]})
                    conversion = pd.concat([conversion,raw], ignore_index=True)
        
                    # print(outname)
                    # stacks shape: (T)CZYX
                    imsave(os.path.join(outpath,outname),stacks, byteorder='>', imagej=True,
                                    metadata={'mode': 'composite'}, extratags=ijtags, check_contrast=False)
                
            conversion.to_csv(os.path.join(outpath, 'metadata.csv'))
            


