import load_data_v1 as ld
import numpy as np
import os

def from_data_to_image(subject_folder):
    import find_segmentation_zone as fsz
    import zone_to_image as zti
    
    data = ld.load_data_for_one_subject(subject_folder)
    zone_combine_list = fsz.find_segmentation_zone_main(data)
    image_zone_list_all =  zti.image_segmentation_data_to_zone(data, zone_combine_list)

def store_image(input_folder, output_folder, image_zone_list_all):
    # if the folder does not exit, make one.
    #get the name for this subject
    file_name_all = os.listdir(subject_folder)
    subject_name = file_name_all[0].split('.')[0]
    
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)    
    
    # save the image to output folder. 
    n_zone_max = 17
    n_view = len(image_zone_list_all)
    zone_count = np.zeros(n_zone_max).astype(np.int32)

    for ii in range(n_view):
        # name for this file.
        for jj in range(n_zone_max):
            if image_zone_list_all[ii][jj] is not None:
                zone_count[jj] = zone_count[jj] + 1
                file_name_this = subject_name + '_zone' + str(jj)+'_'+str(zone_count[jj]) + '.txt'
                file_path_this = os.path.joint(output_folder, file_name_this)
                np.savetxt(file_path_this, image_zone_list_all[ii][jj])

                
def transform_data_to_image(input_folder, output_folder):
    image_zone_list_all = from_data_to_image(input_folder)
    store_image(input_folder, output_folder, image_zone_list_all)