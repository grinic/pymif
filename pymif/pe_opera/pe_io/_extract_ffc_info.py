import xml.dom.minidom
import numpy as np
import os, re

def compute_ff(ff_info, ch_name):
    print("Computing FF profile of channel %s ..."%ch_name)

    x_array = (np.arange(ff_info["profile_dims"][0]) - ff_info["profile_origin"][0])*ff_info["profile_scale"][0]
    y_array = (np.arange(ff_info["profile_dims"][1]) - ff_info["profile_origin"][1])*ff_info["profile_scale"][1]

    x, y = np.meshgrid(x_array, y_array, copy=False)

    c = ff_info['profile_coeffs']

    z = (
        np.zeros(x.shape) + c[0][0]      + 
        c[1][0]*x    + c[1][1]*y      + 
        c[2][0]*x**2 + c[2][1]*x*y    + c[2][2]*y**2      + 
        c[3][0]*x**3 + c[3][1]*x**2*y + c[3][2]*x*y**2    + c[3][3]*y**3   + 
        c[4][0]*x**4 + c[4][1]*x**3*y + c[4][2]*x**2*y**2 + c[4][3]*x*y**3 + c[4][4]*y**4 + 
        0.
    )

    return z

def extract_ffc_info(path, channel_order):

    print('Extracting FFC info from xml file...')

    xml_doc = xml.dom.minidom.parse(os.path.join(path,"Images",'Index.idx.xml'))
    
    entries = xml_doc.getElementsByTagName('Maps')[0].getElementsByTagName('Map')[0].getElementsByTagName('Entry')
    channels = [int(entry.getAttribute('ChannelID')) for entry in entries]
    ffs_info = [entry.getElementsByTagName('FlatfieldProfile')[0].childNodes[0].data for entry in entries]

    ffs_dict = [None for i in channel_order]

    assert len(channel_order) == len(channels), "channel_order must have same length as channel!"

    for i, ch in enumerate(channel_order):
        ch_idx = channels.index(ch+1)
        info = ffs_info[ch_idx]
    #     print(info)
        channel = int(re.findall("Channel: (\d+)", info)[0])
        channel_name = re.findall("ChannelName: (.*), Version", info)[0]
        character = re.findall("Character: (\w*),", info)[0]
        ff_dict = dict(channel = channel,
                       channel_name = channel_name,
                       character = character,
                       ff_profile = 1.
                      )
        # parse data
        if character == 'NonFlat':
            mean = float(re.findall("Mean: (\d+.\d+),", info)[0])
            noiseconst = float(re.findall("NoiseConst: (\d+.\d+),", info)[0])
            nonflatness_corr = float(re.findall("Corrected: (\d+.\d+),", info)[0])
            nonflatness_original = float(re.findall("Original: (\d+.\d+),", info)[0])
            nonflatness_random = float(re.findall("Random: (\d+.\d+)", info)[0])
            coeff_str = re.findall("Coefficients: (.*), Dims", info)[0][1:-1]
            if ", Dims" in coeff_str:
                coeff_str = re.findall("(.*), Dims", coeff_str)[0][:-1]
            profile_coeffs = [[float(a) for a in l.split(", ")] for l in re.findall(r"[^[]*\[([^]]*)\]", coeff_str)]
            dims_str = re.findall(", Dims: (.*), Origin", info)[0][1:-1]
            if ", Origin" in dims_str:
                dims_str = re.findall("(.*), Origin", dims_str)[0][:-1]
            profile_dims = [int(a) for a in dims_str.split(", ")]
            origin_str = re.findall(", Origin: (.*), Scale", info)[0][1:-1]
            if ", Scale" in origin_str:
                origin_str = re.findall("(.*), Scale", origin_str)[0][:-1]
            profile_origin = [float(a) for a in origin_str.split(", ")]
            scale_str = re.findall(", Scale: (.*), Type", info)[0][1:-1]
            if ", Type" in scale_str:
                scale_str = re.findall("(.*), Type", scale_str)[0][:-1]
            profile_scale = [float(a) for a in scale_str.split(", ")]
            ff_dict['mean'] = mean
            ff_dict['noiseconst'] = noiseconst
            ff_dict['nonflatness_corr'] = nonflatness_corr
            ff_dict['nonflatness_original'] = nonflatness_original
            ff_dict['nonflatness_random'] = nonflatness_random
            ff_dict['profile_coeffs'] = profile_coeffs
            ff_dict['profile_dims'] = profile_dims
            ff_dict['profile_origin'] = profile_origin
            ff_dict['profile_scale'] = profile_scale
            ff_dict['ff_profile'] = compute_ff( ff_dict, channel_name )

        ffs_dict[i] = ff_dict

    return ffs_dict

# if __name__ == "__main__":
#     print('Test function')

#     folder = os.path.join('..','data','20230711_laser_align_chroma_widefield__2023-07-11T12_57_48-Measurement 1','FFC_profile')
#     import xml.dom.minidom
#     xml_doc = xml.dom.minidom.parse(os.path.join(folder,'FFC_Profile_Measurement 1.xml'))
#     channel_order = [4,2,0,3,1]
