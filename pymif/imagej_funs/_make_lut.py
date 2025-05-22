import numpy as np
from matplotlib import colors

def make_lut(luts_name, start_color='black'):
    '''
    pass luts_name as lists (e.g. ['red'])
    '''

    luts_dict = {}

    for lut_name in luts_name:
        color_list = [start_color, lut_name]
        my_cmap = colors.LinearSegmentedColormap.from_list("mycmap", color_list)
        a = np.array([ np.array(my_cmap(i)) for i in range(256) ]).T
        a = a[:3]
        a = (a*255).astype(np.uint8)
        luts_dict[lut_name] = np.zeros((3, 256), dtype=np.uint8)
        luts_dict[lut_name] = a

    return luts_dict