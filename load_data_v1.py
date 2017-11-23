import read_img_tsa as tsa_rf
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def list_dir_endswith(path_name, extension_name):
    file_name_all = os.listdir(path_name)
    file_name_correct_extension = []
    for file_name in file_name_all:
        if file_name.endswith(extension_name) and not file_name.startswith('.'):
            file_name_correct_extension.append(file_name)
    
    return file_name_correct_extension
    

def plot_image_movie(data):
    fig = plt.figure(figsize = (16,16))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]
    return animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=200, blit=True)

def create_animation_for_data_path(root_folder, file_name):
    import read_img_tsa as tsa_rf
    file_full_path = os.path.join(root_folder,  file_name)
    data = tsa_rf.read_data(file_full_path)
    animation_object = plot_image_movie(data)
    
    animation_file_name = os.path.join(root_folder, os.path.splitext(file_name)[0] + os.path.splitext(file_name)[1][1:] + '.mp4')
    animation_object.save(animation_file_name, fps=30, extra_args=['-vcodec', 'libx264'])

# # search for all the files in this folder. 3d, or 
# file_extension =['aps','a3daps','a3d','ahi']
# for ff in range(3):
#     file_name_all = list_dir_endswith(root_folder, file_extension[ff])
#     for file_name in file_name_all:
#         create_animation_for_data_path(root_folder, file_name)

def load_data_for_one_subject(root_folder):
#         return file_name_correct_extension
    file_extension =['.a3daps','.a3d','.aps','.ahi']
    data = {}
    for ff in range(2):
        file_name_all = list_dir_endswith(root_folder, file_extension[ff])
        file_full_path = os.path.join(root_folder,  file_name_all[0])
        data[file_extension[ff]] = tsa_rf.read_data(file_full_path)

    return data