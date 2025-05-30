import numpy as np
import glob, os, struct, string, tqdm
from tqdm.utils import _term_move_up
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox
from matplotlib import colors

##########################################################################################

def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))

def make_lut(luts_name):

    luts_dict = {}
    for lut_name in luts_name:
        color_list = ["black", lut_name]
        my_cmap = colors.LinearSegmentedColormap.from_list("mycmap", color_list)
        a = np.array([ np.array(my_cmap(i)) for i in range(256) ]).T
        a = a[:3]
        a = (a*255).astype(np.uint8)
        luts_dict[lut_name] = np.zeros((3, 256), dtype=np.uint8)
        luts_dict[lut_name] = a

    return luts_dict

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

