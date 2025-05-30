import os, tqdm, h5py, glob, json
import numpy as np
from skimage.io import imsave
import pandas as pd

def h52tif_raw(exp_folder, doMIP=True, sigma=1.):

    folder = os.path.join(exp_folder, 'raw')
    print(folder)

    # find all subfolders: all images
    list_subfolders_with_paths = [f.path for f in os.scandir(folder) if f.is_dir()]
    list_subfolders_with_paths.sort()
    print('Found ',len(list_subfolders_with_paths), 'folders.')

    stacks = [path.split('\\')[-1][:path.split('\\')[-1].index('_channel')] for path in list_subfolders_with_paths]
    stacks = list(set(stacks))
    stacks.sort()
    print('Found ',len(stacks), 'stacks:')
    print(stacks)
    
    for stack in stacks:
        print('*'*5,stack,':')
            
        ### find all folders of that stack
        stack_raw_folders = [i for i in list_subfolders_with_paths if stack in i]
        print('\tFound the following folders:')
        print('\t',stack_raw_folders)

        ### find channels for that stack        
        match_start = stack+'_channel_'
        match_end = '_obj_'
        ch_names = [int(i[i.index(match_start)+len(match_start):i.index(match_end)]) for i in stack_raw_folders]#len(stack_raw_folders)//2
        ch_names = list(set(ch_names))
        ch_names.sort()
        print('\tFound the following channels:')
        print('\t',ch_names)
        
        ### find timepoints of that stack
        n_tp = len(glob.glob(os.path.join(stack_raw_folders[0],'*.h5')))
        print('\tNumber of timepoints found:')
        print('\t',n_tp)
        
        for ch in ch_names:
            channel_folders = [i for i in stack_raw_folders if 'channel_%d'%ch in i]
            channel_folders.sort()
            
            
            for tp in range(n_tp):
                print('\tProcessing channel %d, timepoint %d'%(ch,tp))
                
                fnames = []
                imgs = [[],[]]
                i=0
                print('\t\tLoading %d images'%len(channel_folders))
                for direction_folder in channel_folders:
                    fname = glob.glob(os.path.join(direction_folder,'Cam_*_%05d.lux.h5'%tp))[0]
                    fnames.append( fname )
                    f = h5py.File(fname, 'r')
                    imgs[i] = np.array(f['Data']).astype(np.uint16)        
                    i+=1
                # flip the right camera view left to right
                imgs[1] = imgs[1][:,:,::-1]
                
                ### invert the Z if highZ->lowZ acquisition was used
                print('\t\tChecking Z direction from json file...')
                try:
                    metadata = json.load(open(glob.glob(os.path.join(direction_folder,'*.json'))[0]))
                    start_z = metadata['processingInformation']['acquisition'][0]['stage_positions'][2]['start_um']
                    end_z = metadata['processingInformation']['acquisition'][0]['stage_positions'][2]['end_um']
                    if start_z>end_z:
                        imgs[0] = imgs[0][::-1,:,:]
                        imgs[1] = imgs[1][::-1,:,:]
                except:
                    print('\t\tWARNING: could not find json file or something else was wrong. Do not ask.')
                    
                ### DO THE FUSION
                
                ### find out translation for registration
                print('\t\tLoading landmarks and computing sigmoids.')
                p0 = pd.read_csv(os.path.join(exp_folder,'landmarks0.csv'))
                p1 = pd.read_csv(os.path.join(exp_folder,'landmarks1.csv'))

                a = np.array(p0[['X','Y','Slice']])
                b = np.array(p1[['X','Y','Slice']])

                dx = int(np.mean(a-b,0)[0])
                dy = int(np.mean(a-b,0)[1])
                # print('Translation:',dx,dy)

                ### find out sigmoid weights for fusion
                x = np.arange(imgs[0].shape[0])
                X0 = np.max(x)/2.
                r = np.clip((x-X0)/sigma,-1000,1000) # prevent overflow

                sigmoid1 = 1./(1.+np.exp(-r))
                sigmoid2 = 1.-sigmoid1

                # print(sigmoid1.shape)
                # print(x)

                # plt.plot(x,sigmoid1)
                # plt.show()

                # print(t,i)
                # print('reading images channel %d'%i)
                # print('registering images channel %d'%i)
                imgs[1] = np.roll(imgs[1],dy,1)
                imgs[1] = np.roll(imgs[1],dx,2)
                # print('fusing images channel %d'%i)
                # d = a*sigmoid1[:,np.newaxis,np.newaxis]+c*sigmoid2[:,np.newaxis,np.newaxis]
                print('\t\tFusing images.')
                fused_image = imgs[0]*sigmoid1[:,np.newaxis,np.newaxis]+imgs[1]*sigmoid2[:,np.newaxis,np.newaxis]
                
                ### save the fused data
                if not os.path.exists(os.path.join(exp_folder, stack)):
                    os.mkdir(os.path.join(exp_folder,stack))  
                if not os.path.exists(os.path.join(exp_folder, stack, 'fused')):
                    os.mkdir(os.path.join(exp_folder,stack, 'fused'))  
                
                fused_image_name = os.path.join(exp_folder, stack, 'fused', 'tp%03d_ch%d.tif'%(tp,ch))
                imsave(fused_image_name, fused_image.astype(np.uint16), metadata={'axes': 'ZYX'}, check_contrast=False)
                
                ### MIP
                if doMIP:
                    print('\t\tComputing MIP...')

                    mip_outfolder = os.path.join(exp_folder,stack,'fused_mip')
                    if not os.path.exists(mip_outfolder):
                        os.mkdir(mip_outfolder) 
                    
                    def save_max_proj(dset_sub, suffix, ch):
                        ### compute max-projections
                        xymip = np.max(dset_sub,0)
                        yzmip = np.max(dset_sub,1)
                        xzmip = np.transpose(np.max(dset_sub,2))

                        mip = np.concatenate((xymip,yzmip))
                        patch = np.zeros((mip.shape[0]-xzmip.shape[0], xzmip.shape[1]))

                        xzmip = np.concatenate((xzmip, patch))
                        mip = np.concatenate((mip,xzmip),1).astype(np.uint16)

                        filename, fileext = os.path.splitext(fused_image_name.split(os.sep)[-1])

                        new_name = os.path.join(mip_outfolder, filename+'_MIP%s'%(suffix)+fileext)
                        imsave(new_name, mip, check_contrast=False)

                    # save_max_proj(dset[:int(dset.shape[0]/2)], '_front', ch, downscale)
                    # save_max_proj(dset[int(dset.shape[0]/2):], '_back', ch, downscale)
                    save_max_proj(fused_image, '', ch)

                    # print(xymip.shape)
                    # print(yzmip.shape)
                    # print(xzmip.shape)

                    
        


if __name__=='__main__':

            
    master_folders = ['Y:\\Jia_Le_Lim\\data\\2022\\luxendo\\2022-10-25_150411']
            
    for exp_folder in master_folders:        
        h52tif_raw(exp_folder)

