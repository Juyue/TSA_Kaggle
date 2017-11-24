import find_segmentation_zone as fsz
import copy as cp
import numpy as np
def image_segmentation_cut_one_zone_from_one_image(image, image_zone_mask_not, plot_flag = False):
    
    from skimage.filters import threshold_otsu
    from skimage import morphology
    import scipy.ndimage as nd

    image_zone = cp.copy(image)
    image_zone[image_zone_mask_not] = 0

    thresh = threshold_otsu(image_zone)
    binary = image_zone > thresh
    I_mask_image_zone = morphology.remove_small_objects(binary, min_size = 64, connectivity= 2)

    I_change = cp.copy(I_mask_image_zone)
    dilation_size = 20
    struct = np.ones([dilation_size, dilation_size])
    I_change = nd.binary_dilation(I_change, struct)
    I_change = nd.binary_erosion(I_change, struct)

    ## find upper,top, lef, height and width.
    ver_start, ver_end, hor_start, hor_end = fsz.image_segmentation_plot_utils_find_range(I_change)

    ## cut the spefici region.
    image_zone_final = image_zone[ver_start:ver_end, hor_start: hor_end]

    if plot_flag:
        fig, axes = plt.subplots(1,5,figsize=(15,15))
        ax = axes.ravel()
        ax[0].imshow(image)
        ax[1].imshow(image_zone)
        ax[2].imshow(I_mask_image_zone)
        ax[3].imshow(I_change)
        ax[4].imshow(image_zone_final)

        plt.show()
    
    return image_zone_final
def image_segmentation_image_zone_resize(image_zone, zone_num, target_size = (150, 150)):
    from skimage.transform import resize
#     if zone_num == 2:
#         image = cp.copy(image_zone[: ,0: target_size[1]])
#     elif zone_num == 4:
#         nx, ny = image_zone.shape
#         image = cp.copy(image_zone[:, ny - target_size[1]: ny])
#     else:
#         image = cp.copy(image_zone)
    
    image_resized = resize(image_zone, target_size)
    return image_resized

def image_segmentation_I_to_zone(image, zone_mask, target_size = (150, 150)):
    n_zone = zone_mask.shape[2]
    image_zone_list = [None for ii in range(17)]
    for zz in range(17):
        zone_mask_this = zone_mask[:,:,zz]
        if np.sum( zone_mask_this) > 0:
            image_zone_mask_not = np.array( zone_mask_this != True)
            image_zone_original = image_segmentation_cut_one_zone_from_one_image(image, image_zone_mask_not, plot_flag = False)
            image_resize = image_segmentation_image_zone_resize(image_zone_original, zz + 1, target_size = target_size)
            image_zone_list[zz] = image_resize
    return image_zone_list


def image_segmentation_data_to_zone(data, zone_mask_list):
    n_views = data['.a3daps'].shape[2]
    image_zone_list_all = [None for ii in range(n_views)]
    for ii in range(n_views):
        image = cp.copy(np.flipud(data['.a3daps'][:,:,ii].transpose()))
        image_zone_list_all[ii] = image_segmentation_I_to_zone(image, zone_mask_list[ii])
    return image_zone_list_all