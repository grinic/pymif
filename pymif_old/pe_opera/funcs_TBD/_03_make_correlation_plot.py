import numpy as np
from skimage.io import imread, imsave
import glob, os, tqdm
from imagej_fun import imagej_metadata_tags, make_lut
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_uint
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################

# def binary2correlation(
#                 path, 
#             ):

#     flist = glob.glob(os.path.join(path,'*bin-2-2-1--C*'))
#     flist.sort()
#     N = len(flist)


#     file_name = os.path.basename(flist[0])
#     file_path = os.path.dirname(flist[0])
#     file_root, file_ext = os.path.splitext(file_name)

#     org_mask = imread(os.path.join(file_path,'binary',file_root+'_0.tif'))
#     V_gastr = float(np.sum(org_mask))

#     volumes = np.array([0. for i in flist])
#     overlap = np.zeros((N,N))

#     masks = [0 for i in flist]
#     i = 0
#     for file in flist:
#         file_name = os.path.basename(file)
#         file_path = os.path.dirname(file)
#         file_root, file_ext = os.path.splitext(file_name)

#         masks[i] = imread(os.path.join(file_path,'binary',file_name)).astype(float)
#         v = float(np.sum(masks[i]))

#         # print(v/V_gastr)
#         volumes[i] = v/V_gastr
#         i+=1

#     for i in range(N):
#         for j in range(N):
#             m = masks[i] * masks[j]
#             v = np.sum(m)
#             overlap[i,j] = v/V_gastr

#     V_tot_df = pd.DataFrame({'V_tot':[V_gastr]})
#     V_channel_df = pd.DataFrame({'V_channel':volumes})
#     V_channel_df.columns = ['Volume_fraction']
#     V_channel_df.index = ['ch%d'%i for i in range(N)]
#     V_overlap_df = pd.DataFrame(overlap)
#     V_overlap_df.columns = ['ch%d'%i for i in range(N)]
#     V_overlap_df.index = ['ch%d'%i for i in range(N)]

#     writer = pd.ExcelWriter(os.path.join(file_path,'binary','volumes.xlsx'), engine='xlsxwriter')
#     V_tot_df.to_excel(writer, sheet_name='V_tot_absolute')
#     V_channel_df.to_excel(writer, sheet_name='V_channel_fraction')
#     V_overlap_df.to_excel(writer, sheet_name='V_overlap_fraction')
#     writer.save()

# '''
# To read, use:
# pd.read_excel('volumes.xlsx',sheet_name='V_overlap_fraction',header=0,index_col=0).to_numpy()
# '''

# ################################################################################

# if __name__=='__main__':

#     paths = [ 
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_114830'),
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_115427'),
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_120041'),
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_121905'),
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_122616'),
#                 os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
#                     '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_123405'),
#             ]

#     for path in tqdm.tqdm(paths):
#         binary2volume(path)


