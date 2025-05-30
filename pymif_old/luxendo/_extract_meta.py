import os, glob
import json
import h5py

def extract_meta(path):
    sample_meta = {}

    infos = path.split(os.sep)[-1]
    sample_meta["folder"] = infos

    raw_folders = [f.split(os.sep)[-1] for f in glob.glob(os.path.join(path,"raw","*"))]
    raw_folders.sort()
    # print(raw_folders)
    channels = [int(p.split("channel_")[-1][0]) for p in raw_folders]
    # print(channels)
    n_ch = max(channels)+1
    sample_meta["n_ch"] = n_ch
    # print(n_ch)
    # print(infos)

    for i in range(sample_meta["n_ch"]):
        json_folder = glob.glob(os.path.join(path,"raw","*channel_%d*"%i))
        json_folder.sort()
        json_file = glob.glob(os.path.join(json_folder[0], "*.json"))
        json_file.sort()
        json_file=json_file[0]

        sample_meta["ch-%d_index"%i] = i

        with open(json_file) as f:
            data = json.load(f)
        if "channel_description" in data["processingInformation"]["acquisition"][0].keys():
            sample_meta["ch-%d_name"%i] = data["processingInformation"]["acquisition"][0]["channel_description"]
        else:
            sample_meta["ch-%d_name"%i] = "channel%d"%i

        sample_meta["ch-%d_wavelength"%i] = [l["wavelength_nm"] for l in data["metaData"]["lasers"] if l["name"]!="LED" and l["on"]][0]
        sample_meta["ch-%d_intensity"%i] = [l["intensity"] for l in data["metaData"]["lasers"] if l["name"]!="LED" and l["on"]][0]

    meta_file = glob.glob(os.path.join(path, "*.xml"))[0]

    import xml.dom.minidom
    xml_doc = xml.dom.minidom.parse(meta_file)
    unit = xml_doc.getElementsByTagName("SpimData")[0].getElementsByTagName("ViewSetups")[0].getElementsByTagName("ViewSetup")[0].getElementsByTagName("voxelSize")[0].getElementsByTagName("unit")[0].childNodes[0].data
    size = xml_doc.getElementsByTagName("SpimData")[0].getElementsByTagName("ViewSetups")[0].getElementsByTagName("ViewSetup")[0].getElementsByTagName("voxelSize")[0].getElementsByTagName('size')[0].childNodes[0].data

    sample_meta["scale_1_1_1_unit"] = unit
    sample_meta["scale_1_1_1_z"] = [float(s) for s in size.split(" ")][2]
    sample_meta["scale_1_1_1_y"] = [float(s) for s in size.split(" ")][1]
    sample_meta["scale_1_1_1_x"] = [float(s) for s in size.split(" ")][0]

    return sample_meta

def test_extract_meta():
    path = os.path.join("/g","ebisuya","input-tray","Cell_count","data","primary",
                        "MC20230609_MM2_750_DAPI-405_SOX2-488_TBX6-594_BRA-647",
                        "2023-09-06_193223")
    sample_meta = extract_meta(path)
    print(sample_meta)

if __name__=="__main__":
    test_extract_meta()
