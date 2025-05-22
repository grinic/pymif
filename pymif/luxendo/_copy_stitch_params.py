import os, glob
import shutil

def copy_stitch_params(exp_folders):

    n = 0
    for f in exp_folders:
        print(n, '-----', f)

        ### figure out number of channels
        raw_folder = os.path.join('D:',os.sep,'Users','Marina_Matsumiya',f.split('\\')[0], 'raw')
        names_acq = os.listdir(raw_folder)
        ch_numbers = [int(name_acq[name_acq.index('channel_')+8]) for name_acq in names_acq]
        n_ch = 0
        for ch_number in ch_numbers:
            if ch_number>n_ch:
                n_ch = ch_number
        n_ch += 1

        #### reg_obj
        reg = 'reg_obj'

        folder = os.path.join('D:',os.sep,'Users','Marina_Matsumiya', f, reg)

        filenames = glob.glob(os.path.join(folder,'tp-0_ch-*.json'))

        if len(filenames)==1:

            print('Copying', reg)

            filename = filenames[0]
            current_ch = int(filename[filename.index('tp-0_ch-')+8])

            for i in range(n_ch):
                newname = filename.replace('tp-0_ch-%d_st-0'%current_ch,'tp-0_ch-%d_st-0'%i)
                if i != current_ch:
                    shutil.copy(filename, newname)

        #### reg_sti
        # figure out number of tiles
        n_acq = len(os.listdir(raw_folder))
        n_tiles = n_acq//2//n_ch
        
        reg = 'reg_sti'

        folder = os.path.join('D:',os.sep,'Users','Marina_Matsumiya', f, reg)

        filenames = glob.glob(os.path.join(folder,'tp-0_ch-*.json'))

        if (len(filenames)==(n_tiles*(n_tiles-1)//2))&(len(filenames)!=0):

            print('Copying', reg)

            for filename in filenames:

                current_ch = int(filename[filename.index('tp-0_ch-')+8])

                for i in range(n_ch):
                    newname = filename.replace('tp-0_ch-%d_st-0'%current_ch,'tp-0_ch-%d_st-0'%i)
                    if i!=current_ch:
                        shutil.copy(filename, newname)


        n += 1

##################################################################################

if __name__=='__main__':

    folders = [
            '2021-05-05_151536-10uM-mat\\processed\\20220823-100937_Task_1_Description_for_C',
            '2021-05-20_162527-7uM\\processed\\20220823-102955_Task_1_Description_for_C',
            '2021-06-05_143725-8uM\\processed\\20220224-151207_Task_14_Description_for_C',
            '2021-06-05_144135-8uM\\processed\\20220224-151323_Task_15_Description_for_C',
            '2021-06-05_150919-9uM\\processed\\20220224-151444_Task_16_Description_for_C',
            '2021-06-05_151911-10uM\\processed\\20220224-151601_Task_17_Description_for_C',
            '2021-06-05_152432-7uM\\processed\\20220224-151706_Task_18_Description_for_C',
            '2021-06-05_152829-7uM\\processed\\20220224-151825_Task_19_Description_for_C',
            '2021-06-05_153238-7uM\\processed\\20220224-151959_Task_20_Description_for_C',
            '2021-06-05_153802-6uM\\processed\\20220224-152105_Task_21_Description_for_C',
            '2021-06-05_154157-6uM\\processed\\20220224-152234_Task_22_Description_for_C',
            '2021-06-05_154556-6uM\\processed\\20220224-152450_Task_23_Description_for_C',
            '2021-06-05_155117-5uM\\processed\\20220224-152601_Task_24_Description_for_C',
            '2021-06-05_155450-5uM\\processed\\20220224-152713_Task_25_Description_for_C',
            '2021-06-05_155832-5uM\\processed\\20220224-152848_Task_26_Description_for_C',
            '2021-06-23_143658-nomat\\processed\\20220223-172320_Task_5_Description_for_C',
    ]

    copy_stitch_params(folders)
