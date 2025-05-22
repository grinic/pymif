import pandas as pd
import xml.etree.ElementTree as et 
import tqdm, os

#####################

def xml2csv(exp_folder, save=False):

    xtree = et.parse(os.path.join(exp_folder,'Index.idx.xml'))
    xroot = xtree.getroot()
    # print(xroot.attrib['Images'])
    images = xroot.findall('{http://www.perkinelmer.com/PEHH/HarmonyV5}Images')[0]
    print(len(images))

    df = pd.DataFrame({'filename':[], 
                        'Xpos':[], 'Ypos':[], 'Zpos':[], 
                        'row':[], 'col':[], 
                        'field':[], 'plane':[], 
                        'channel':[], 'chName':[], 
                        'expTime':[]})

    for image in tqdm.tqdm(images.iter('{http://www.perkinelmer.com/PEHH/HarmonyV5}Image'), total=len(images)):
        # print(image.tag, image.attrib)

        row = {}
        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}URL')
        row['filename'] = x.text

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionX')
        row['Xpos'] = float(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionY')
        row['Ypos'] = float(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionZ')
        row['Zpos'] = float(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}Row')
        row['row'] = int(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}Col')
        row['col'] = int(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}FieldID')
        row['field'] = int(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PlaneID')
        row['plane'] = int(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelID')
        row['channel'] = int(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelName')
        row['chName'] = x.text

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ExposureTime')
        row['expTime'] = float(x.text)

        x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}TimepointID')
        row['Timepoint'] = int(x.text)
        
        df = df.append(pd.Series(row), ignore_index=True)

    print(df.head())

    if save:
        df.to_csv(os.path.join(os.path.split(exp_folder)[0],'metadata.csv'))
    return df
