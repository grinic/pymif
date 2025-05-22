import os, tqdm, h5py, glob
import numpy as np
from skimage.io import imread, imsave

def h52tif_fused(infolder, outfolder, downscale=1, doMIP=True, identifier='uni'):

    print(infolder)

    # find all h5 files: all images
    image_list = glob.glob(os.path.join(infolder,identifier+'*.h5'))
    print('Found ',len(image_list), 'images with', identifier, 'identifier.')
    
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    for fname in tqdm.tqdm(image_list):
        f = h5py.File(fname, 'r')
        dset = np.array(f['Data']).astype(np.uint16)

        # extract channel name and angle
        ch = int(fname[fname.index('ch-')+3])

        dset = dset[:,::downscale,::downscale]
        # print(dset.shape)

        # create name of new file
        new_name = os.path.join(outfolder,'ch-%d_bin-%d%d1.tif'%(ch,downscale,downscale))

        # save image as tif file
        imsave(new_name, dset.astype(np.uint16), check_contrast=False)

    ### MIP
    if doMIP:
        print('Computing MIP...')

        flist = glob.glob(os.path.join(outfolder,'ch*_bin-%d%d1.tif'%(downscale,downscale)))

        ch = 0
        for fname in tqdm.tqdm(flist):

            dset = imread(fname)

            ### compute max-projections
            xymip = np.max(dset,0)
            yzmip = np.max(dset,1)
            xzmip = np.transpose(np.max(dset,2))

            mip = np.concatenate((xymip,yzmip))
            patch = np.zeros((mip.shape[0]-xzmip.shape[0], xzmip.shape[1]))

            xzmip = np.concatenate((xzmip, patch))
            mip = np.concatenate((mip,xzmip),1).astype(np.uint16)

            new_name = os.path.join(outfolder,'ch-%d_bin-%d%d1_MIP.tif'%(ch,downscale,downscale))
            imsave(new_name, mip, check_contrast=False)

            # print(xymip.shape)
            # print(yzmip.shape)
            # print(xzmip.shape)

            ch += 1

if __name__=='__main__':

    for input_folder, output_folder in zip(input_folders, output_folders):
        h52tif_fused(input_folder, output_folder, downscale)
