import numpy as np
from skimage.io import imread, imsave
import glob, os, string, tqdm
from ...imagej_funs._make_lut import make_lut
from ...imagej_funs._imagej_metadata_tags import imagej_metadata_tags

def compile_timelapse(path, channel_order, luts_name, dT=1, **kwargs):
    '''This function combines timelapse images of a 96WP acquired by PE.

    Parameters
    ----------
    path: string
            a string containing the path t the experiment folder. Has to point to the "Images" folder
    channels_order: iterable, int
            a list, same length of paths. Each element contains the order in which the channels have to be arranged in the output images.
            E.g. if PE acquires GFP first and BF next, and you want BF-GFP, then the order is [1,0]
    luts_name: iterable, string
            list of colors to be used to show the channels
    dT: int, optional, default: 1
            process every dT image

    Returns
    -------
    a "compiled" folder, containing the well subfolders, containing 1 multichannel tif file for every timepoint in the experiment

    NOTE:
    This script assume the experiment contains just one FOV per well!
    '''
    # find all tiff files in the folder
    flist = glob.glob(os.path.join(path,'*.tiff'))
    flist.sort()

    # find out all positions (the first 6 characters, e.g.: r01c01 )
    pos = list(set( [os.path.split(f)[-1][:6] for f in flist] ))
    pos.sort()

    # define well id to convert e.g. r01c01 into A01
    d = dict(enumerate(string.ascii_uppercase, 1))

    wells_list = []
    if 'pos_list' not in kwargs.keys():
        for p in pos:
            wells_list.append( d[int(p[1:3])]+p[4:6] )
    else:
        wells_list = kwargs['pos_list']

    pos_process = []
    for p in pos:
        well = d[int(p[1:3])]+p[4:6]
        if well in wells_list:
            pos_process.append(p)

    pbar = tqdm.tqdm(pos_process)
    for p in pbar:
        well = d[int(p[1:3])]+p[4:6]

        pbar.set_description(p + ' ' + well)
        pbar.update()
        
        outpath = os.path.join(os.path.split(path)[0],'compiled',well)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

            # extract all files from this well
            flist = glob.glob(os.path.join(path,p+'*.tiff'))
            flist.sort()

            # find timepoints
            timepoints = list(set([f[f.index('sk')+2:f.index('fk')] for f in flist]))
            timepoints = np.array([int(t) for t in timepoints])
            timepoints.sort()
            timepoints = timepoints[::dT]

            # for each timepoint
            for timepoint in tqdm.tqdm(timepoints):

                # extract all files from this timepoint
                channels_list = glob.glob(os.path.join(path,p+'*'+'sk'+str(timepoint)+'fk*.tiff'))
                channels_list.sort()

                # find channels
                channels = list(set([f[f.index('-ch')+3:f.index('-ch')+4] for f in channels_list]))
                channels = np.array([int(ch) for ch in channels])
                channels.sort()

                stacks = []
                # for each channel
                for channel in channels:
                    # extract all files from this channel
                    stack_list = glob.glob(os.path.join(path,p+'*-ch'+str(channel)+'*sk'+str(timepoint)+'fk*.tiff'))
                    stack_list.sort()

                    stack = []
                    # create 3D stack
                    for f in stack_list:
                        stack.append(imread(f))
                    stack = np.array(stack)
                    # if just 2D, drop a dimension
                    if stack.shape[0]==1:
                        stack = stack[0]

                    # append the channel stack to the multichannel array
                    stacks.append(stack)

                stacks = np.array(stacks).astype(np.uint16)

                # order channels according to input
                stacks = np.array([stacks[ch] for ch in channel_order]).astype(np.uint16)

                if stacks.ndim==4:
                    stacks = np.swapaxes(stacks,0,1)
                
                # create imagej metadata with LUTs
                luts_dict = make_lut(luts_name)
                ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')

                # create filename
                outname = well+'_tp%05d.tif'%(timepoint-1)

                # save array
                imsave(os.path.join(outpath,outname),stacks, byteorder='>', imagej=True,
                                metadata={'mode': 'composite'}, extratags=ijtags)

##########################################################################################

if __name__=='__main__':

    paths = [   
                os.path.join(
                    '2021-02-09_braGFP_2i_96-120hpa_TL',
                    'Images'
                    ),
            ]

    # this is to, for instance, arrange BF in first channel and GFP in second channel
    # available LUTS: gray, red, green, blue, magenta, cyan, yellow
    channel_orders = [ 
            [1,0] 
        ]
    luts_names = [ 
            ['gray', 'green'] 
        ]

    # use >1 to comile fewer images (e.g. for testing)
    dT = 1

    # select only certain wells:
    remove_list = []

    pos_list = [
            [l+'%02d'%i for i in range(1,13) for l in string.ascii_uppercase[:8] ]
        ]
    i=0
    for rm in remove_list:
        for remove in rm:
            pos_list[i].remove(remove)
        i+=1
    print(pos_list)


    for i in range(len(paths)):
        path = paths[i]
        channel_order = channel_orders[i]
        luts_name = luts_names[i]

        compile_timelapse(path, channel_order, luts_name, dT, pos_list=pos_list[0])
