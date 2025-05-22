import os, glob
import numpy as np
from skimage.io import imread, imsave

from ..imagej_funs._make_lut import make_lut
from ..image_preprocessing._normalize import normalize

def tif2png(
                path, 
                luts_name = ['gray','green','cyan','magenta','yellow','red'], 
                percs = [[3,97],[3,97],[3,97],[3,97],[3,97],[3,97]]
            ):

    parent = os.path.dirname(path)
    print(parent)
    if not os.path.exists(os.path.join(parent,'pngs')):
        os.mkdir(os.path.join(parent,'pngs'))

    # print(os.path.join(path,'*_MIP.tif'))
    flist = glob.glob(os.path.join(path,'*_MIP.tif'))
    flist.sort()
    downscale = int(flist[0][flist[0].index('bin')+3])
    # print(flist)

    composite = []

    i = 0
    for _file in flist:
        perc = percs[i]

        img = imread(_file).astype(float)
        img = normalize(img, perc)

        # percs_val = np.percentile(img, tuple(perc))
        # img = (img-percs_val[0])/(percs_val[1]-percs_val[0])
        # img = np.clip(img, 0 , 1)
        ### convert to uint8 and use all dynamic range
        img = 255*img
        img = img.astype(np.uint8)
        
        # print(pos)
        file_name = os.path.basename(_file)
        file_path = os.path.dirname(_file)
        file_root, file_ext = os.path.splitext(file_name)
        
        new_file = os.path.join(parent,'pngs',file_root+'.png')
        
        imsave(new_file, img)

        composite.append(img)

        i+=1

    composite = np.array(composite).astype(np.uint8)
    # new_file = os.path.join(path,'pngs','composite.tif')

    luts_dict = make_lut(luts_name)
    # ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')

    # imsave(new_file, composite, byteorder='>', imagej=True,
    #         metadata={'mode': 'composite'}, extratags=ijtags)

    ### convert composite tif into png rgb
    comp_png = np.zeros((img.shape[0],img.shape[1],3)).astype(np.float)
    n = 0
    for c in composite:
        i=0
        for row in c:
            j=0
            for val in row:
                rgb = luts_dict[luts_name[n]][:,val]
                comp_png[i,j,:] += rgb
                j+=1
            i+=1
        n+=1
    comp_png = (np.array(comp_png)-np.min(comp_png))/(np.max(comp_png)-np.min(comp_png))
    comp_png = np.clip(comp_png, 0 , 1)
    comp_png = 255*comp_png
    comp_png = comp_png.astype(np.uint8)
    new_file = os.path.join(parent,'pngs','composite_bin%d%d1.png'%(downscale,downscale))
    imsave(new_file,comp_png)

#####################################################################

if __name__=='__main__':

    import tqdm
    folders = ''

    paths = [os.path.join(i,'fused') for i in folders]

    luts = [
        'green',    # Bra
        'magenta',  # Meox1
        'yellow',   # Sox2
        'gray'      # Foxa2
        ]
    percs = [
        [90,99],
        [80,99],
        [80,99],
        [90,99]
        ]

    for path in tqdm.tqdm(paths):

        tif2png(path,luts,percs)
