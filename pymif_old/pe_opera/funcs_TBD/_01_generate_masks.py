import numpy as np
from skimage.io import imread, imsave
import gc
import glob, os, tqdm
from imagej_fun import imagej_metadata_tags, make_lut
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_uint, img_as_ubyte

###############################################################################

def tif2binary(
                path, 
            ):

    flist = glob.glob(os.path.join(path,'ch*.tif'))
    downscale = int(flist[0][flist[0].index('bin')+3])

    flist = glob.glob(os.path.join(path,'ch*_bin%d%d1.tif'%(downscale,downscale)))
    flist.sort()

    # load all images
    print('Reading channels...')
    imgs = np.array([imread(f).astype(int) for f in tqdm.tqdm(flist)])

    # create full organoid mask
    print('Computing threshold...')
    sumimg = np.sum(imgs,0)
    thr = threshold_otsu(sumimg[::2,::2,::2])
    org_mask = img_as_uint(sumimg>thr)

    # keep largest object, from:
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    print('Keeping largest objects...')
    labels = label(org_mask)
    org_mask = labels == np.argmax(np.bincount(labels.flat, weights=org_mask.flat))

    print('Save file...')
    file_name = os.path.basename(flist[0])
    file_path = os.path.dirname(flist[0])
    file_root, file_ext = os.path.splitext(file_name)
    if not os.path.exists(os.path.join(file_path,'binary')):
        os.mkdir(os.path.join(file_path,'binary'))
    new_file = os.path.join(file_path,'binary',file_root+'_0.tif')
    imsave(new_file, org_mask, check_contrast=False)

    del(sumimg)
    gc.collect()
    
    imgs = np.array([i.astype(np.uint16) for i in imgs])
    i = 0
    print('Make fluorescence masks...')
    for img, file in tqdm.tqdm(zip(imgs, flist)):

        percs_val = np.percentile(img[:,::2,::2], (3,99))
        img = (img-percs_val[0])/(percs_val[1]-percs_val[0])
        img = np.clip(img, 0 , 1)
        img = 255*img
        img = img.astype(np.uint8)

        # compute threshold within the gastruloid
        fluo_gastr = img.flatten()[org_mask.flatten()]
        thr = threshold_otsu(fluo_gastr)
        mask = img>thr
        # remove objects detected outside the gastruloid
        mask = mask*org_mask
        
        # print(pos)
        file_name = os.path.basename(file)
        file_path = os.path.dirname(file)
        file_root, file_ext = os.path.splitext(file_name)
        
        if not os.path.exists(os.path.join(file_path,'binary')):
            os.mkdir(os.path.join(file_path,'binary'))
        new_file = os.path.join(file_path,'binary',file_root+'.tif')
        
        imsave(new_file, mask, check_contrast=False)

        i+=1
        
        gc.collect()

################################################################################

if __name__=='__main__':

    paths = [ 
                # os.path.join(
                #     'Y:',os.sep,'Nicola_Gritti','raw_data',
                #     '2020-10-16_gastrHCR','Cond1_72h','2020-10-16_101422','processed'),
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

    for path in tqdm.tqdm(paths):
        tif2binary(path)
    