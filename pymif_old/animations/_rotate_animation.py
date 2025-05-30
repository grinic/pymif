from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def rotate_animation_df(df, col_names, ncols=4, elev=15, azim_init=-60, 
            figsize=(15,15), save=False, interval=10, folder='', show_ticks=False):
    """
    

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    col_names : list, str
        should contain a valid column value.
    ncols : int
    elev : float
    azim : float
    figsize : tuple
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # check that all keys provided are valid
    assert len(col_names)>0
    assert all([i in df.keys() for i in col_names])
    
    
    fig = plt.figure(figsize=figsize)
    axs = []
    nrows = (len(col_names)-1)//ncols + 1
    
    i = 0
    for col_name in col_names:
        ax = fig.add_subplot(nrows,ncols,i+1,projection='3d')
        ax.scatter(
            df.x_unit, df.y_unit, df.z_unit, 
            c = df[col_name], 
            cmap = 'plasma',
            alpha=0.2
           )
            
        ax.title.set_text(col_name)
        if not show_ticks:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
        axs.append(ax)
        
        i+=1
        
    plt.tight_layout()
        
    def update(frame):
        for ax in axs:
            ax.view_init(elev=elev, azim=azim_init+frame)
    
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 365, 36), interval=interval)
    
    return ani