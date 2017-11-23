import skimage as im
import numpy as np
import matplotlib.pyplot as plt

def create_mask_from_image(image, plot_flag = False):
    import matplotlib
    import numpy as np

    from skimage import img_as_float
    from skimage import exposure

    from skimage.filters import threshold_otsu

    from skimage import morphology

    thresh = threshold_otsu(image)
    binary = image > thresh
    I_mask = morphology.remove_small_objects(binary, min_size = 64, connectivity= 2)

    if plot_flag:
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(binary, cmap=plt.cm.gray)
        ax[1].set_title('Thresholded')
        ax[1].axis('off')

        ax[2].imshow(I_mask , cmap=plt.cm.gray)
        ax[2].set_title('cleaned ')
        ax[2].axis('off')

        plt.show()
    
    return I_mask

def find_matlab_style(x, n = None, mode = 'all' ):
    non_zero_idx = [ii for (ii, val) in enumerate(x) if val != False]
    if non_zeros_idx is None
        return None
#     bp()
    if mode.startswith('all'):
        return non_zero_idx
    elif mode.startswith('first'):
        return non_zero_idx[0:n]
    elif mode.startswith('last'):
        return non_zero_idx[-(n + 1):-1]
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def find_extreme_in_a_region(x, mask, mode = None):
    import copy as cp
    x_use = cp.copy(x)
    if mode.startswith('max'):
        x_use[mask == 0] = -1e3
#         bp()
        sort_value = np.sort(x_use, axis = None)
        sort_pos = np.argsort(x_use, axis = None)
        val_selected = int(sort_value[-1])
        pos_selected = int(sort_pos[-1])
    elif mode.startswith('min'):
        x_use[mask == 0] = 1e3;
#         bp()
        sort_value = np.sort(x_use, axis = None).astype(np.int64)
        sort_pos = np.argsort(x_use, axis = None).astype(np.int64)
        val_selected = int(sort_value[0])
        pos_selected = int(sort_pos[0])
    else:
        val_selected = None
        pos_selected = None

    return val_selected, pos_selected



# write a function, use mask function to find the lines.
def find_segmentation_line_0(I):
    import numpy as np
    n_points = 5;
    n_smooth = 10;
    wid_brain = 30;
    
    (n_x, n_y) = I.shape
    
    top_position = np.zeros(n_y)
    bottom_position = np.zeros(n_y)
    left_position = np.zeros(n_x)
    right_position = np.zeros(n_x)
    for yy in range(n_y):
        this_vertical_line = I[:, yy]
        if np.sum(this_vertical_line) > n_points:
            top_position[yy] = np.mean(find_matlab_style(this_vertical_line, n_points, 'first'))
            bottom_position[yy] = np.mean(find_matlab_style(this_vertical_line, n_points , 'last'))

    for xx in range(n_x):
        this_horizontal_line = I[xx, :]
        if np.sum(this_horizontal_line) > n_points :
            left_position[xx] =  np.mean(find_matlab_style(this_horizontal_line, n_points, 'first'))
            right_position[xx] =  np.mean(find_matlab_style(this_horizontal_line, n_points, 'last'))
    
    top_position = smooth(top_position, n_smooth)
    bottom_position = smooth(bottom_position, n_smooth)
    left_position = smooth(left_position, n_smooth)
    right_position = smooth(right_position, n_smooth)
    body_contour = right_position - left_position
    
    
#     fig = plt.figure(figsize = (4,4))
#     plt.subplot(3,2,1)
#     plt.plot(top_position)
#     plt.subplot(3,2,2)
#     plt.plot(bottom_position);
#     plt.subplot(3,2,3)
#     plt.plot(left_position);
#     plt.subplot(3,2,4)
#     plt.plot(right_position);
#     plt.subplot(3,2,5)
#     plt.plot(bottom_position - top_position);
#     plt.subplot(3,2,6)
#     plt.plot(right_position - left_position);
#     plt.show()
    
    mid_y = int(np.floor(n_y/2));
    parts_horizontal = np.array([[0, mid_y], [mid_y + 1, n_y - 1], [mid_y - wid_brain, mid_y + wid_brain] ], dtype = np.int64)
    mid_x = int(np.floor(n_x/2));
    parts_vertical = np.array([[0, mid_x ], [mid_x  + 1, n_x - 1]], dtype = np.int64)

    left_elbow = np.zeros(2).astype(np.int64)
    right_elbow = np.zeros(2).astype(np.int64)

        
    left_image_mask = np.zeros(n_y).astype(np.int64)
    left_image_mask[parts_horizontal[0,0]:parts_horizontal[0,1]] = 1
    left_elbow[0],  left_elbow[1] =  find_extreme_in_a_region(top_position, left_image_mask, 'max');

    right_image_mask = np.zeros(n_y).astype(np.int64)
    right_image_mask[parts_horizontal[1,0]:parts_horizontal[1,1]] = 1;
    right_elbow[0],  right_elbow[1] =  find_extreme_in_a_region(top_position, right_image_mask, 'max');

    middle_image_mask = np.zeros(n_y).astype(np.int64)
    middle_image_mask[parts_horizontal[2,0]:parts_horizontal[2,1]] = 1;
    
    top_position_diff = np.concatenate((np.zeros(1), np.diff(top_position)))

    dummy, left_hand_region_y = find_extreme_in_a_region(top_position_diff, middle_image_mask, 'max');
    dummy, right_hand_region_y = find_extreme_in_a_region(top_position_diff, middle_image_mask, 'min');
    
    brain_height = np.max(top_position[left_hand_region_y : right_hand_region_y]).astype(np.int64)

    leg_hip_joint = np.zeros(2).astype(np.int64)
    leg_hip_joint[0],leg_hip_joint[1]= find_extreme_in_a_region(bottom_position, middle_image_mask, 'min');
    
    foot_position = int(np.max(bottom_position))
    
    ## use total height and leg length to estimate body parts, and segment image horizontally
    leg_length = foot_position - leg_hip_joint[0]
    total_height = foot_position - brain_height
    upper_body_length = total_height - leg_length
    neck_position = np.floor(brain_height + upper_body_length/5).astype(np.int64)
    chest_position = np.floor(brain_height + upper_body_length/5 * 2).astype(np.int64)
    waist_position = np.floor(brain_height + upper_body_length/5 * 3.75).astype(np.int64)
    middle_thigh_position = np.floor(foot_position - leg_length/5 * 4).astype(np.int64)
    upper_keen_position =  np.floor(foot_position -leg_length/4 * 2).astype(np.int64)
    lower_keen_position = np.floor(foot_position - leg_length/4).astype(np.int64)
    
    
    ## segment the image vertically...
    body_middle_line = np.mean([leg_hip_joint[1], left_elbow[1], right_elbow[1], mid_y]).astype(np.int64)
    groin_half_width = 35
    left_groin = body_middle_line - groin_half_width
    right_groin = body_middle_line + groin_half_width
    trunk_width = np.mean(body_contour[chest_position: waist_position])
    left_trunk = np.floor(body_middle_line - trunk_width/2).astype(np.int64)
    right_trunk = np.floor(body_middle_line + trunk_width/2).astype(np.int64)
    
    segmentation_hor = {'left_elbow' : left_elbow[0],
                        'right_elbow': right_elbow[0],
                        'head' : brain_height,
                        'neck' : neck_position,
                        'chest': chest_position,
                        'waist': waist_position,
                        'thigh': middle_thigh_position,
                        'knee_upper' : upper_keen_position,
                        'knee_lower' : lower_keen_position,
                        'foot':foot_position,
                        'groin' : leg_hip_joint[0]
                       }
    segmentation_ver = {'left_elbow' : left_elbow[1],
                        'right_elbow': right_elbow[1],
                        'left_trunk' :left_trunk ,
                        'right_trunk' :right_trunk ,
                        'middle':body_middle_line,
                        'left_groin' :left_groin,
                        'right_groin' : right_groin ,
                       }
    segmentation = {'hor':segmentation_hor,
                   'ver':segmentation_ver}
    return segmentation

def plot_segmentation_and_body(I, segmentation):
    seg_hor = segmentation['hor']
    seg_ver = segmentation['ver']
    (n_x, n_y) = I.shape
    plt.imshow(I)

    x_lim = [0,n_y];
    
    plt.plot(x_lim, [seg_hor['left_elbow'],seg_hor['left_elbow']], 'k-')
    plt.plot(x_lim, [seg_hor['head'],seg_hor['head']], 'k-')
    plt.plot(x_lim, [seg_hor['neck'],seg_hor['neck']],'k-')
    plt.plot(x_lim, [seg_hor['chest'],seg_hor['chest']],'k-')
    plt.plot(x_lim, [seg_hor['waist'],seg_hor['waist']],'k-')
    plt.plot(x_lim, [seg_hor['thigh'],seg_hor['thigh']],'k-')
    plt.plot(x_lim, [seg_hor['knee_upper'],seg_hor['knee_upper']],'k-')
    plt.plot(x_lim, [seg_hor['knee_lower'],seg_hor['knee_lower']],'k-')
    plt.plot(x_lim, [seg_hor['foot'],seg_hor['foot']],'k-');
    y_lim = [0,n_x];

    plt.plot([seg_ver['left_elbow'], seg_ver['left_elbow']], y_lim, 'k-');
    plt.plot([seg_ver['left_trunk'], seg_ver['left_trunk']], y_lim, 'k-');
    plt.plot([seg_ver['left_groin'], seg_ver['left_groin']], y_lim, 'k-');

    plt.plot([seg_ver['middle'], seg_ver['middle']], y_lim, 'k-');
    plt.plot([seg_ver['right_groin'], seg_ver['right_groin']], y_lim, 'k-')
    plt.plot([seg_ver['right_trunk'], seg_ver['right_trunk']], y_lim, 'k-');
    plt.plot([seg_ver['right_elbow'], seg_ver['right_elbow']], y_lim, 'k-');
#     plt.subplot(3, 2, 6)
#     imagesc(zone_mask_combine)
    plt.show()
    
def front_image_segmentation_line_to_zone(segmentation, I_mask):
    n_zone = 17;
    (n_x, n_y) = I_mask.shape
    seg_hor = segmentation['hor']
    seg_ver = segmentation['ver']
    middle_of_elbow = int((seg_hor['left_elbow'] + seg_hor['right_elbow'])/2);
    
    zone_mask = np.zeros((n_x, n_y, n_zone)) == 1
    
    zone_mask[0                     : seg_hor['left_elbow'],  0                     : seg_ver['middle'],     1] = True
    zone_mask[seg_hor['left_elbow'] : seg_hor['chest'],       0                     : seg_ver['left_trunk'], 0] = True

    zone_mask[0                     : seg_hor['right_elbow'], seg_ver['middle']     :n_y - 1,                3] = True
    zone_mask[seg_hor['right_elbow']: seg_hor['chest'],       seg_ver['right_trunk']:n_y - 1,                2] = True

    zone_mask[middle_of_elbow       : seg_hor['chest'],       seg_ver['left_trunk'] : seg_ver['right_trunk'], 4] = True

    zone_mask[seg_hor['chest']      : seg_hor['waist'],       0                     : seg_ver['middle']  ,    5]= True
    zone_mask[seg_hor['chest']      : seg_hor['waist'],       seg_ver['middle']     : n_y - 1  ,              6]= True
                         
    zone_mask[seg_hor['waist']      : seg_hor['thigh'],       seg_ver['left_trunk'] : seg_ver['left_groin'],  7]= True
    zone_mask[seg_hor['waist']      : seg_hor['thigh'],       seg_ver['left_groin'] : seg_ver['right_groin'],8]= True 
    zone_mask[seg_hor['waist']      : seg_hor['thigh'],       seg_ver['right_groin']:seg_ver['right_trunk'] ,9]= True   
                         
                         
    zone_mask[seg_hor['thigh']      : seg_hor['knee_upper'],  0                     :seg_ver['middle'] ,10]= True                    
    zone_mask[seg_hor['thigh']      : seg_hor['knee_upper'],  seg_ver['middle']     :n_y - 1 ,          11]= True      
                         
    zone_mask[seg_hor['knee_upper'] : seg_hor['knee_lower'],  0                     :seg_ver['middle'] ,12]= True                    
    zone_mask[seg_hor['knee_upper'] : seg_hor['knee_lower'],  seg_ver['middle']     :n_y - 1 ,          13]= True   
    
    zone_mask[seg_hor['knee_lower'] : seg_hor['foot'],        0                     :seg_ver['middle'] ,14]= True                    
    zone_mask[seg_hor['knee_lower'] : seg_hor['foot'],        seg_ver['middle']     :n_y - 1 ,          15]= True 
                         
    zone_mask_combine = np.zeros((n_x, n_y));
    for zz in range(n_zone):
        zone_mask_combine[zone_mask[:,:,zz] & I_mask] = zz;
    
    return zone_mask_combine
