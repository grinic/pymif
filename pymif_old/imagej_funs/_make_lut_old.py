import numpy as np

def make_lut_old():
    # generate LUT for primary and secondary colors

    # Intensity value range
    val_range = np.arange(256, dtype=np.uint8)
    luts_dict = {}
    # Gray LUT
    luts_dict['gray'] = np.stack([val_range, val_range, val_range])
    # Red LUT
    luts_dict['red'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['red'][0, :] = val_range
    # Green LUT
    luts_dict['green'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['green'][1, :] = val_range
    # Blue LUT
    luts_dict['blue'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['blue'][2, :] = val_range
    # Magenta LUT
    luts_dict['magenta'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['magenta'][0, :] = val_range
    luts_dict['magenta'][2, :] = val_range
    # Cyan LUT
    luts_dict['cyan'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['cyan'][1, :] = val_range
    luts_dict['cyan'][2, :] = val_range
    # Yellow LUT
    luts_dict['yellow'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['yellow'][0, :] = val_range
    luts_dict['yellow'][1, :] = val_range
    # Orange LUT
    luts_dict['orange'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['orange'][0, :] = val_range
    luts_dict['orange'][1, :] = (165.*val_range/256.).astype(np.uint8)
    # Maroon LUT
    luts_dict['maroon'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['maroon'][0, :] = (128.*val_range/256.).astype(np.uint8)

    return luts_dict
