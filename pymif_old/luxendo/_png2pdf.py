import glob, os, tqdm
from skimage.io import imread
import matplotlib.pyplot as plt

from matplotlib import rc
# from matplotlib.colors import LinearSegmentedColormap
rc('font', size=18)
rc('font', family='Arial')
rc('pdf', fonttype=42)
plt.rcParams.update({'font.size': 15})

def png2pdf(
                inpaths, 
                outpath,
                cond,
                gene
            ):

    imgs = [imread(fname) for fname in glob.glob(os.path.join(inpaths[0],'ch*.png'))]
    nrows=len(inpaths)
    ncols=len(imgs)+1
    
    fig,ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols,2*nrows))
    fig.subplots_adjust(right=0.95,bottom=0.05,wspace=0.01,hspace=0.1)
    fig.suptitle(cond)

    for i, inpath in enumerate(inpaths):
        imgs = [imread(fname) for fname in glob.glob(os.path.join(inpath,'ch*.png'))]
        for j, img in enumerate(imgs):
            ax[i,j].imshow(img, cmap='gray')
        comp = imread(os.path.join(inpath,'composite_bin221.png'))
        ax[i,-1].imshow(comp)
        
    for a in ax.flatten():
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])

    for a, g in zip(ax[0,:-1],gene):
        a.title.set_text(g)
    for a, inpath in zip(ax[:,0], inpaths):
        a.set_ylabel(inpath,fontsize=5)
        
    fig.savefig(os.path.join(outpath,cond+'.pdf'), dpi=300)


if __name__=='__main__':

    conditions = [
        'control',
        'control24h',
        'pd03',
        'sb43',
        'xav'
        ]

    input_folders = [[] for i in conditions]
    output_folders = []
    genes = [['Bra','Sox17','Sox2','Foxa2'] for i in conditions]
    for i, cond in enumerate(conditions):
        exps = [f.path for f in os.scandir(os.path.join(cond)) if f.is_dir()]
        output_folders.append(os.path.dirname(os.path.join(cond)))
        for exp in exps:
            if os.path.exists(os.path.join(exp,'pngs')):
                input_folders[i].append(os.path.join(exp,'pngs'))

    for input_folder, output_folder, cond, gene in tqdm.tqdm(zip(input_folders,output_folders, conditions, genes)):
        png2pdf(input_folder,output_folder,cond, gene)
