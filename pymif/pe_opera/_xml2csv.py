import pandas as pd
import xml.etree.ElementTree as et
import tqdm
import os, glob


def xml2csv(exp_folder,
            image_folder = "Images",
            meta_file_name = "metadata.csv",
            save = True):

    # print(os.path.join(exp_folder, image_folder, "*.xml"))
    xml_file = glob.glob(os.path.join(exp_folder, image_folder, "*.xml"))[0]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()

    if len(xroot.findall("{http://www.perkinelmer.com/PEHH/HarmonyV5}Images"))>0:
        df = xml2csv_v5(exp_folder,
            image_folder = image_folder,
            meta_file_name = meta_file_name,
            save = save)
    elif len(xroot.findall("{http://www.perkinelmer.com/PEHH/HarmonyV6}Images"))>0:
        df = xml2csv_v6(exp_folder,
            image_folder = image_folder,
            meta_file_name = meta_file_name,
            save = save)
    else:
        raise ValueError("Harmony version not recognized!")
            
    return df
    
def xml2csv_v5(exp_folder,
            image_folder = "Images",
            meta_file_name = "metadata.csv",
            save = True):

    # print(os.path.join(exp_folder, image_folder, "*.xml"))
    xml_file = glob.glob(os.path.join(exp_folder, image_folder, "*.xml"))[0]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()

    images = xroot.findall("{http://www.perkinelmer.com/PEHH/HarmonyV5}Images")[0]
    print("Found %d images."%len(images))

    df = pd.DataFrame(
        {
            "filename": [],
            "Xpos": [],
            "Ypos": [],
            "Zpos": [],
            "row": [],
            "col": [],
            "field": [],
            "plane": [],
            "channel": [],
            "chName": [],
            "expTime": [],
        }
    )


    for i, image in tqdm.tqdm(enumerate(images.iter("{http://www.perkinelmer.com/PEHH/HarmonyV5}Image")), total=len(images)):
        # print(image.tag, image.attrib)

        row = {}
        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}URL")
        row["filename"] = x.text

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionX")
        row["Xpos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionY")
        row["Ypos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionZ")
        row["Zpos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}Row")
        row["row"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}Col")
        row["col"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}FieldID")
        row["field"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}PlaneID")
        row["plane"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}TimepointID")
        row["timepoint"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelID")
        row["channel"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelName")
        row["chName"] = x.text

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelType")
        row["chType"] = x.text

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}MainExcitationWavelength")
        row["chWavelength"] = x.text

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}ExposureTime")
        row["expTime"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}ImageResolutionX")
        row["pixelSize"] = float(x.text)*1e6

        df = pd.concat([df, pd.Series(row).to_frame().T], ignore_index=True)


    # print(df.head())
    if save:
        df.to_csv(os.path.join(exp_folder, meta_file_name))

    return df

def xml2csv_v6(exp_folder,
            image_folder = "Images",
            meta_file_name = "metadata.csv",
            save = True):
    version = "{http://www.perkinelmer.com/PEHH/HarmonyV6}"

    # print(os.path.join(exp_folder, image_folder, "*.xml"))
    xml_file = glob.glob(os.path.join(exp_folder, image_folder, "*.xml"))[0]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()

    channels = xroot.findall(version+"Maps")[0]
    print("Found %d channels."%len(channels))
    print(channels)
    print(channels.findall(version+"Map"))
    
    # for country in root.findall('country'):
    #     rank = country.find('rank').text
    #     name = country.get('name')
    #     print(name, rank)
        
    df_ch = pd.DataFrame({
        "channel": [],
        "chName": [],
        "acquisitionType": [],
        "chType": [],
        "pixelSize": [],
        "binning": [],
        "chWavelength" :[],
        "chWavelengthEmission": [],
        "objMagnification": [],
        "objNA": [],
        "expTime": [],
        "excitationPower": []
    })
    for channel in channels.findall(version+"Map"):
        for ch in channel.findall(version+"Entry"):
            print(ch.tag, ch.attrib)
            print(ch.get("ChannelID"))
            if len(ch.findall(version+"ChannelName"))>0: 
                row = {}           
                print("ChannelName:",ch.find(version+"ChannelName").text)
                row["channel"] = ch.get("ChannelID")
                row["chName"] = ch.find(version+"ChannelName").text
                row["acquisitionType"] = ch.find(version+"AcquisitionType").text
                row["chType"] = ch.find(version+"ChannelType").text
                row["pixelSize"] = float(ch.find(version+"ImageResolutionX").text)*1e6
                row["binning"] = int(ch.find(version+"BinningX").text)
                row["chWavelength"] = int(ch.find(version+"MainExcitationWavelength").text)
                row["chWavelengthEmission"] = int(ch.find(version+"MainEmissionWavelength").text)
                row["objMagnification"] = ch.find(version+"ObjectiveMagnification").text
                row["objNA"] = ch.find(version+"ObjectiveNA").text
                row["expTime"] = float(ch.find(version+"ExposureTime").text)
                row["excitationPower"] = float(ch.find(version+"ExcitationPower").text)
                
                df_ch = pd.concat([df_ch, pd.Series(row).to_frame().T], ignore_index=True)

    images = xroot.findall(version+"Images")[0]
    print("Found %d images."%len(images))

    df = pd.DataFrame(
        {
            "filename": [],
            "Xpos": [],
            "Ypos": [],
            "Zpos": [],
            "row": [],
            "col": [],
            "field": [],
            "plane": [],
            "channel": [],
            "chName": [],
            "expTime": [],
        }
    )
    
    for i, image in tqdm.tqdm(enumerate(images.iter("{http://www.perkinelmer.com/PEHH/HarmonyV6}Image")), total=len(images)):
        # print(image.tag, image.attrib)

        row = {}
        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}URL")
        row["filename"] = x.text

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}PositionX")
        row["Xpos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}PositionY")
        row["Ypos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}PositionZ")
        row["Zpos"] = float(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}Row")
        row["row"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}Col")
        row["col"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}FieldID")
        row["field"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}PlaneID")
        row["plane"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}TimepointID")
        row["timepoint"] = int(x.text)

        x = image.find("{http://www.perkinelmer.com/PEHH/HarmonyV6}ChannelID")
        # row["channel"] = int(x.text)
        
        channel_df = dict(df_ch[df_ch.channel==x.text].squeeze(axis=0))
        # print(channel_df)
        # print(row)
        
        for el in channel_df.keys():
            # print(el, channel_df[el])
            row[el] = channel_df[el]

        df = pd.concat([df, pd.Series(row).to_frame().T], ignore_index=True)


    # print(df.head())
    if save:
        df.to_csv(os.path.join(exp_folder, meta_file_name))

    return df

#####################################################################################

if __name__ == '__main__':
    #####################

    ### windows nicola
    # exp_folder = "/g/trivedi/Kristina_Stapornwongkul/ImageAnalysis/gastr_hcr_volumes/data/primary/date-20220304_hpa-96_plate-1_exp-1"
    exp_folder = "PATH-TO-EXPERIMENT"

    xml2csv(exp_folder,
        image_folder = "Images",
        meta_file_name = "metadata_PE.csv",
        save = True) 

    #####################

