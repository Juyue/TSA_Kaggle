def load_one_image():
    import read_img_tsa as tsa_rf
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation

    root_folder = r'E:\Juyue\Kaggle_Data\sample\sample'

    def list_dir_endswith(path_name, extension_name):
        file_name_all = os.listdir(path_name)
        file_name_correct_extension = []
        for file_name in file_name_all:
            if file_name.endswith(extension_name) and not file_name.startswith('.'):
                file_name_correct_extension.append(file_name)

        return file_name_correct_extension
    file_extension =['.aps','.a3daps','.a3d','.ahi']

    ff = 1
    file_name_all = list_dir_endswith(root_folder, file_extension[ff])
    data = []
    file_full_path = os.path.join(root_folder,  file_name_all[0])
    data.append(tsa_rf.read_data(file_full_path))

    d_this = data[0]
    # I = np.flipud(d_this[:,:,0].transpose())
    return d_this