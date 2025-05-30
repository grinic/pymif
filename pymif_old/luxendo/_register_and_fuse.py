from skimage.io import imread, imsave
import numpy as np
import pandas as pd
import os, glob, tqdm
'''
Assumes only 1 tile!
'''

def register_and_fuse(exp_folder, doFusion=True, doMIP=True, sigma=1., acquisition_direction='lowZ->highZ'):

    print('Processing ',folder)

    infolder = os.path.join(exp_folder,'raw')

    ### Register and fuse images
    outfolder = os.path.join(exp_folder,'fused')
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    downscale = glob.glob(os.path.join(infolder,'ch-*_x00-y00_obj-left_bin-*1.tif'))[0]
    downscale = int(downscale[downscale.index('bin')+4])
    # print(downscale)

    ### FUSION
    if doFusion:
        print('Fusing opposing views...')

        ### find out translation for registration
        p = pd.read_csv(os.path.join(infolder,'landmarks.csv'))

        a = p.iloc[::2]
        a = np.array(a[['X','Y','Slice']])
        b = p.iloc[1::2]
        b = np.array(b[['X','Y','Slice']])

        dx = int(np.mean(a-b,0)[0])
        dy = int(np.mean(a-b,0)[1])
        # print('Translation:',dx,dy)


        ### find out sigmoid weights for fusion
        a = imread(os.path.join(infolder,'ch-0_x00-y00_obj-left_bin-%d%d1.tif'%(downscale,downscale)))

        x = np.arange(a.shape[0])
        X0 = np.max(x)/2.
        r = np.clip((x-X0)/sigma,-1000,1000) # prevent overflow

        sigmoid1 = 1./(1.+np.exp(-r))
        sigmoid2 = 1.-sigmoid1

        # print(sigmoid1.shape)
        # print(x)

        # plt.plot(x,sigmoid1)
        # plt.show()

        n_ch = len(glob.glob(os.path.join(infolder,'ch-*_obj-left*.tif')))

        for i in tqdm.tqdm(range(n_ch)):
            # print('reading images channel %d'%i)
            a = imread(os.path.join(infolder,'ch-%d_x00-y00_obj-left*.tif'%i))
            b = imread(os.path.join(infolder,'ch-%d_x00-y00_obj-right*.tif'%i))
            # print('registering images channel %d'%i)
            c = np.roll(b,dy,1)
            c = np.roll(c,dx,2)
            # print('fusing images channel %d'%i)
            # d = a*sigmoid1[:,np.newaxis,np.newaxis]+c*sigmoid2[:,np.newaxis,np.newaxis]
            if acquisition_direction=='lowZ->highZ':
                d = a*sigmoid1[:,np.newaxis,np.newaxis]+c*sigmoid2[:,np.newaxis,np.newaxis]
            elif acquisition_direction=='highZ->lowZ':
                d = a*sigmoid2[:,np.newaxis,np.newaxis]+c*sigmoid1[:,np.newaxis,np.newaxis]
            imsave(os.path.join(outfolder,'ch-%d_x00-y00_bin-%d%d1.tif'%(i,downscale,downscale)), d.astype(np.uint16), metadata={'axes': 'ZYX'}, check_contrast=False)
            # os.remove(os.path.join(master_folder,'ch%d_ang000.tif'%i))
            # os.remove(os.path.join(master_folder,'ch%d_ang180.tif'%i))

    ### MIP
    if doMIP:
        print('Computing MIP...')

        flist = glob.glob(os.path.join(outfolder,'ch-*_bin-%d%d1.tif'%(downscale,downscale)))

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

            new_name = os.path.join(outfolder,'ch-%d_x00-y00_bin-%d%d1_MIP.tif'%(ch,downscale,downscale))
            imsave(new_name, mip, check_contrast=False)

            # print(xymip.shape)
            # print(yzmip.shape)
            # print(xzmip.shape)

            ch += 1

######################################

if __name__=='__main__':
    
    folders = [
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_091057'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_092015'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_092705'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_093150'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_093401'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_093622'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_094137'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_094927'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_095141'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_095552'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_095848'),
        # os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-15_100146'),
        os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-17_084237'),
        os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-17_084818'),
        os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-17_085709'),
        os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','2021-02-15_gastrHCR','Panel1','2021-02-17_090628'),
    ]

    doFusion = True
    doMIP = True

    for folder in folders:

        fuseAndMIP_folder(folder, doFusion, doMIP)

