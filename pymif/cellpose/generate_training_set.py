import os, glob
from skimage.io import imread, imsave
from tqdm import tqdm
import numpy as np
import pandas as pd

df = pd.read_csv('experiments_list.csv', converters={'processed': lambda x: True if x == 'True' else False}  )


def process_exp(experiment):
    print(experiment)
    
    flist = glob.glob(os.path.join(experiment,'tifs','*.tif'))
    flist.sort()
    ch0 = imread(flist[0])
    n_planes = ch0.shape[0]
    
    n_extracted = n_planes//10
    
    planes_idx = np.arange(n_planes)
    np.random.shuffle(planes_idx)
    planes_extracted = planes_idx[:n_extracted]
    
    for plane in planes_extracted:
        fname = os.path.split(experiment)[-1][:-1]+'_plane%04d.tif'%plane
        imsave(fname, ch0[plane], check_contrast=False)
    
    

# print(df)
experiments_to_process = []
for i,row in df.iterrows():
    if not row.processed:
        experiments_to_process.append(row.experiment)

print(experiments_to_process)

for experiment in tqdm(experiments_to_process):
    process_exp(experiment)
    
df.processed = 'True'

df.to_csv('experiments_list.csv', index=None)

