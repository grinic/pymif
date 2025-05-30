# @Author: Giovanni Dalmasso <gio>
# @Date:   10-Aug-2021
# @Email:  giovanni.dalmasso@embl.es
# @Project: FeARLesS
# @Filename: 01_figurePositions.py
# @Last modified by:   gio
# @Last modified time: 11-Aug-2021
# @License: MIT


import pandas as pd
import os
import tqdm
import struct
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from skimage.io import imread, imsave

####################################


# =============================================================================
# ### mac gio
# # path = "/Volumes/sharpe/data/Vascular_micromass/Opera/TIMELAPSE/" "Timelapse4_041021/"
# # folder_raw = os.path.join(path)
# =============================================================================

# ### linux gio
path = "/g/sharpe/data/Vascular_micromass/Opera/TIMELAPSE/" "Timelapse4_041021/"
folder_raw = os.path.join(path)

# =============================================================================
# ### windows nicola
# path = os.path.join(
#     "data", "Vascular_micromass", "Opera", "TIMELAPSE", "Timelapse4_041021"
# )
# folder_raw = os.path.join("X:", os.sep, path)
# folder_raw = os.path.join('')
# exp_folder = os.path.join(
#     "meta"
# )
# =============================================================================

exp_folder = os.path.join(
    "gio_Pecam-Sox9_20x-24h_041021__2021-10-04T16_06_44-Measurement_1"
)

# name of wells/slides - correspond to columns row/col in the metadata.csv file
well_names = ["well1","well2"]

### how many samples there are in the same well (in the row and col direction)
# should be the same size as slide_names
n_cols = [
    1, # well1
    1  # well2
    ]
n_rows = [
    1, # well1
    1  # well2
    ]

# name of samples in every well
samplesNames = [
    ["A01"], # well1
    ['A02']  # well2
    ]

channel_list = [2, 1] # same for all the wells
luts_name = ["gray", "green"] # same for all the wells

####################################


def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{">": b"IJIJ", "<": b"JIJI"}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode("utf-16" + {">": "be", "<": "le"}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder + ("d" * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ("Info", b"info", 1, writestring),
        ("Labels", b"labl", None, writestring),
        ("Ranges", b"rang", 1, writedoubles),
        ("LUTs", b"luts", None, writebytes),
        ("Plot", b"plot", 1, writebytes),
        ("ROI", b"roi ", 1, writebytes),
        ("Overlays", b"over", None, writebytes),
    )

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == "<":
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder + "I", count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b"".join(body)
    header = b"".join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ("I" * len(bytecounts)), *bytecounts)
    return (
        (50839, "B", len(data), data, True),
        (50838, "I", len(bytecounts) // 4, bytecounts, True),
    )


def make_lut():
    # generate LUT for primary and secondary colors

    # Intensity value range
    val_range = np.arange(256, dtype=np.uint8)
    luts_dict = {}
    # Gray LUT
    luts_dict["gray"] = np.stack([val_range, val_range, val_range])
    # Red LUT
    luts_dict["red"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["red"][0, :] = val_range
    # Green LUT
    luts_dict["green"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["green"][1, :] = val_range
    # Blue LUT
    luts_dict["blue"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["blue"][2, :] = val_range
    # Magenta LUT
    luts_dict["magenta"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["magenta"][0, :] = val_range
    luts_dict["magenta"][2, :] = val_range
    # Cyan LUT
    luts_dict["cyan"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["cyan"][1, :] = val_range
    luts_dict["cyan"][2, :] = val_range
    # Yellow LUT
    luts_dict["yellow"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["yellow"][0, :] = val_range
    luts_dict["yellow"][1, :] = val_range
    # Orange LUT
    luts_dict["orange"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["orange"][0, :] = val_range
    luts_dict["orange"][1, :] = (165.0 * val_range / 256.0).astype(np.uint8)
    # Maroon LUT
    luts_dict["maroon"] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict["maroon"][0, :] = (128.0 * val_range / 256.0).astype(np.uint8)

    return luts_dict


########################################


df = pd.read_csv(os.path.join(folder_raw, exp_folder, "metadata1.csv"))
print(df.head())

wells = df.drop_duplicates(subset=['row','col'])
wells = [[i.row,i.col] for j,i in wells.iterrows()]

s = 0
for well, well_name in zip(wells, well_names):
    df_well = df[(df.row == well[0])&(df.col == well[1])]

    xpos = list(set(df_well.Xpos))
    xpos.sort()
    ypos = list(set(df_well.Ypos))
    ypos.sort()

    if n_cols[s] == 1:
        xlims = [
            np.min(xpos),
            np.max(xpos),
        ]
    elif n_cols[s] == 2:
        xlims = [
            np.min(xpos),
            (np.min(xpos) + np.max(xpos)) / 2,
            np.max(xpos),
        ]
    elif n_cols[s] == 3:
        xlims = [
            np.min(xpos),
            (np.min(xpos) + (np.min(xpos) + np.max(xpos)) / 2) / 2,
            (np.min(xpos) + np.max(xpos)) / 2,
            (np.max(xpos) + (np.min(xpos) + np.max(xpos)) / 2) / 2,
            np.max(xpos),
        ]

    if n_rows[s] == 1:
        ylims = [
            np.min(ypos),
            np.max(ypos),
        ]
    elif n_rows[s] == 2:
        ylims = [
            np.min(ypos),
            (np.min(ypos) + np.max(ypos)) / 2,
            np.max(ypos),
        ]
    elif n_rows[s] == 3:
        ylims = [
            np.min(ypos),
            (np.min(ypos) + (np.min(ypos) + np.max(ypos)) / 2) / 2,
            (np.min(ypos) + np.max(ypos)) / 2,
            (np.max(ypos) + (np.min(ypos) + np.max(ypos)) / 2) / 2,
            np.max(ypos),
        ]


    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(
    #     df_well[df_well.channel == 1].Xpos,
    #     df_well[df_well.channel == 1].Ypos,
    #     c=df_well[df_well.channel == 1]["Unnamed: 0"],
    #     s=10,
    # )
    # output = np.array(list(product(xlims, ylims)))
    # ax.plot(output[:, 0], output[:, 1], "or")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # plt.show()
    
    
    for i in tqdm.tqdm(range(len(xlims) - 1)):
        for j in range(len(ylims) - 1):
    
            print(i, j)
            outFolder = os.path.join(
                folder_raw, exp_folder, well_name + "_" + samplesNames[i][j]
            )
            if not os.path.exists(outFolder):
                os.mkdir(outFolder)
    
            xmin = xlims[i]
            xmax = xlims[i + 1]
            ymin = ylims[j]
            ymax = ylims[j + 1]
    
            df_sample = df_well[(df_well.Xpos <= xmax) & (df_well.Xpos >= xmin)]
            df_sample = df_sample[(df_sample.Ypos <= ymax) & (df_sample.Ypos >= ymin)]
    
            # fig, ax = plt.subplots(1, 1)
            # # ax.set_title(folderName)
            # ax.scatter(
            #     df_poc[df_poc.channel == 1].Xpos,
            #     df_poc[df_poc.channel == 1].Ypos,
            #     c=df_poc[df_poc.channel == 1]["Unnamed: 0"],
            #     s=10,
            # )
            # ax.plot(output[:, 0], output[:, 1], "or")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # plt.show()
    
            xpos_sample = list(set(df_sample.Xpos))
            xpos_sample.sort()
            ypos_sample = list(set(df_sample.Ypos))
            ypos_sample.sort()
    
            # df_poc_ordered = df_poc.sort_values(by=["Xpos","Ypos"])
            # print(df_poc_ordered)
    
            for xidx, x in enumerate(xpos_sample):
                for yidx, y in enumerate(ypos_sample):
                    df_pos = df_sample[(df_sample.Xpos == x) & (df_sample.Ypos == y)]
                    if len(df_pos) > 0:
                        if not os.path.exists(
                            os.path.join(outFolder, "r%03d_c%03d" % (yidx, xidx))
                        ):
                            os.mkdir(os.path.join(outFolder, "r%03d_c%03d" % (yidx, xidx)))
                        # print(len(df_pos))
                        tps = list(set(df_pos.timepoint))
                        tps.sort()
                        for tp in tps:
    
                            df_tp = df_pos[df_pos.timepoint == tp]
                            
                            stack = []
                            for ch in channel_list:
                                df_tp_ch = df_tp[df_tp.channel == ch]
                                df_tp_ch = df_tp_ch.sort_values(by="Zpos")
                                # [print(img_file) for img_file in df_pos_ch.filename]

                                # print(df_tp_ch.filename)
                                
                                stack_ch = np.stack(
                                    [
                                        imread(
                                            os.path.join(
                                                folder_raw, exp_folder, "Images", img_file
                                            )
                                        )
                                        for img_file in df_tp_ch.filename
                                    ]
                                )
                                stack.append(stack_ch)
    
                            # order channels
                            stacks = np.array(stack).astype(np.uint16)
                            stacks = np.swapaxes(stacks, 0, 1)
    
                            # create imagej metadata with LUTs
                            luts_dict = make_lut()
    
                            ijtags = imagej_metadata_tags(
                                {"LUTs": [luts_dict[i] for i in luts_name]}, ">"
                            )
    
                            outname = "r%03d_c%03d_tp%04d.tif" % (yidx, xidx, tp)
                            imsave(
                                os.path.join(
                                    outFolder, "r%03d_c%03d" % (yidx, xidx), outname
                                ),
                                stacks,
                                byteorder=">",
                                imagej=True,
                                metadata={"mode": "composite"},
                                extratags=ijtags,
                            )
    s+=1
    

print("cioa")
