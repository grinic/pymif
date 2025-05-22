# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:29:59 2022

@author: gritti
"""

import sys, os
import pandas as pd
from skimage.io import imread
sys.path.append('W:\\people\\gritti\\code\\pymif')
from pymif import pe_opera

exp_folder = 'Y:\\Nick Marschlich\\EMBL_Barcelona\\Imaging\\Opera PE\\P3_Metabolism\\220929_mezzo-GFP_BF_2-DG_injections'
# ff_folder = 'Y:\\Nicola_Gritti\hannah\\20220824_HF_HBMEC-RM-exp2_FF__2022-08-24T15_45_50-Measurement 1'

'''
First, create the csv file for image bookeeping
'''
pe_opera.xml2csv(exp_folder,
        image_folder = os.path.join("Images"),
        meta_file_name = "metadata_PE.csv",
        save = True)

'''
Next, create the FF images with ImageJ manually....

Finally, compile experimental images using flat field correction
'''
df = pd.read_csv(os.path.join(exp_folder,'metadata_PE.csv'))
channel_order = pe_opera.pe_io.extract_channel_order(df) # [1,0]

pe_opera.wellplate.compile_conditions_multifields_timelapse(
        path = exp_folder, 
        channel_order = channel_order, 
        luts_name = ['white', 'lime'], 
        df = df,
        downsample=1.,
        ff_mode = "PE",
        image_folder = os.path.join("Images"),
)
