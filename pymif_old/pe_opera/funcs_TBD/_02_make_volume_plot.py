import numpy as np
from skimage.io import imread, imsave
import glob, os, tqdm
from imagej_fun import imagej_metadata_tags, make_lut
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_uint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################

def make_volume_plot_plt(paths, ch_names, ch_ordered, ax, dx=0):

    volumes = [[] for i in set(ch_ordered)]

    i = 0
    for path in tqdm.tqdm(paths):
        vol_file = os.path.join(path,'binary','volumes.xlsx')
        v = pd.read_excel(vol_file, sheet_name='V_channel_fraction',header=0,index_col=0).to_numpy()[:,0]

        j = 0
        for ch in ch_names[i]:
            ch_idx = ch_ordered.index(ch)
            volumes[ch_idx].append( float(v[j]) )
            j += 1
        i += 1

    x = np.arange(len(ch_ordered))+1
    print(x+dx, volumes)
    ax.violinplot(volumes, positions = x+dx)
    ax.set_xticks(x)
    ax.set_xticklabels(ch_ordered)


def make_volume_plot_sns(paths, ch_names, ch_ordered, ax, 
                            cond=None,split=False):

    volumes = pd.DataFrame({'sample':[],
                            'volume':[],
                            'gene':[],
                            'cond':[]})

    hue = 'cond'
    if cond == None:
        hue = None
        cond = [None for i in paths]

    i = 0
    for path in tqdm.tqdm(paths):
        vol_file = os.path.join(path,'binary','volumes.xlsx')
        v = pd.read_excel(vol_file, sheet_name='V_channel_fraction',header=0,index_col=0).to_numpy()[:,0]

        j = 0
        for ch in ch_names[i]:
            row = pd.Series({'sample': path,
                            'gene': ch,
                            'volume': float(v[j]),
                            'cond': cond[i]})
            volumes = volumes.append(row, ignore_index=True)
            j += 1
        i += 1

    sns.violinplot(y='volume', x='gene', data=volumes, ax=ax, scale='width', order=ch_ordered,hue=hue,split=split, inner="stick")


'''
To read, use:
pd.read_excel('volumes.xlsx',sheet_name='V_overlap_fraction',header=0,index_col=0).to_numpy()
'''

################################################################################

if __name__=='__main__':

    paths = [ 
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_114830'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_115427'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond1_2I','2020-09-23_120041'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_121905'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_122616'),
                os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data',
                    '2020-09-23_gastrHCR','Cond2_2I','2020-09-23_123405'),
            ]

    ch_names = [
                ['T-GFP','Aldh1a2','Meox1','BMP4'],
                ['T-GFP','Aldh1a2','Meox1','BMP4'],
                ['T-GFP','Aldh1a2','Meox1','BMP4'],
                ['T-GFP','Foxa2','Prdm1','T'],
                ['T-GFP','Foxa2','Prdm1','T'],
                ['T-GFP','Foxa2','Prdm1','T'],
            ]
    
    ch_ordered = ['T-GFP','T','Aldh1a2','Meox1','BMP4','Foxa2','Prdm1']

    # fig1, ax1 = plt.subplots()
    # make_volume_plot_plt(paths, ch_names, ch_ordered, ax1)
    # ax1.set_ylim((0,1))

    fig2, ax2 = plt.subplots()
    make_volume_plot_sns(paths, ch_names, ch_ordered, ax2)
    ax2.set_ylim((0,1))
    ax2.set_xlabel('')
    ax2.set_ylabel('Volume Fraction')

    plt.show()


