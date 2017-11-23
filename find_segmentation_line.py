import load_data_v1 as ld
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
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

def find_segmentation_hor(I, plot_flag):
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
    # should be first peak
#     left_elbow[0],  left_elbow[1] =  find_extreme_in_a_region(top_position, left_image_mask, 'max')
    left_elbow[1] =  np.mean(find_matlab_style(top_position> 50, n_points, 'first'))
    left_elbow[0] = top_position[left_elbow[1].astype(np.int64) + 10]
#     print(top_position[left_elbow[1].astype(np.int64) + 5:left_elbow[1].astype(np.int64) + 20])
    
    right_image_mask = np.zeros(n_y).astype(np.int64)
    right_image_mask[parts_horizontal[1,0]:parts_horizontal[1,1]] = 1
    # should be last peak
#     right_elbow[0],  right_elbow[1] =  find_extreme_in_a_region(top_position, right_image_mask, 'max');
    right_elbow[1] =  np.mean(find_matlab_style(top_position > 50, n_points, 'last'));
    right_elbow[0]  = top_position[right_elbow[1].astype(np.int64) - 10]
    
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
    
    lower_thigh_position = np.floor((middle_thigh_position + upper_keen_position)/2).astype(np.int64)
    keen_position = np.floor((upper_keen_position + lower_keen_position)/2).astype(np.int64)
    middle_calf_position = np.floor((lower_keen_position + foot_position)/2).astype(np.int64)
    
    
    seg_hor = {'left_elbow' : left_elbow[0],
                        'right_elbow': right_elbow[0],
                        'head' : brain_height,
                        'neck' : neck_position,
                        'chest': chest_position,
                        'waist': waist_position,
                        'thigh': middle_thigh_position,
                        'knee_upper' : upper_keen_position,
                        'knee_lower' : lower_keen_position,
                        'foot':foot_position,
                        'lower_thigh':lower_thigh_position,
                        'knee':keen_position,
                        'middle_calf': middle_calf_position,
                        'groin' : leg_hip_joint[0]
                       }
    
    if plot_flag:
        find_segmentation_hor_utils_plotline(I, seg_hor)

#     plt.plot([seg_ver['left_elbow'], seg_ver['left_elbow']], y_lim, 'k-');
#     plt.plot([seg_ver['left_trunk'], seg_ver['left_trunk']], y_lim, 'k-');
#     plt.plot([seg_ver['left_groin'], seg_ver['left_groin']], y_lim, 'k-');

#     plt.plot([seg_ver['middle'], seg_ver['middle']], y_lim, 'k-');
#     plt.plot([seg_ver['right_groin'], seg_ver['right_groin']], y_lim, 'k-')
#     plt.plot([seg_ver['right_trunk'], seg_ver['right_trunk']], y_lim, 'k-');
#     plt.plot([seg_ver['right_elbow'], seg_ver['right_elbow']], y_lim, 'k-');
#     plt.subplot(3, 2, 6)
#     imagesc(zone_mask_combine)
        plt.show()
    
    return seg_hor
def find_segmentation_hor_utils_plotline(I, seg_hor):
    (n_x, n_y) = I.shape
    x_lim = [0,n_y];
    plt.imshow(I)
    plt.plot(x_lim, [seg_hor['left_elbow'],seg_hor['left_elbow']], 'k-')
    plt.plot(x_lim, [seg_hor['head'],seg_hor['head']], 'k-')
    plt.plot(x_lim, [seg_hor['neck'],seg_hor['neck']],'k-')
    plt.plot(x_lim, [seg_hor['chest'],seg_hor['chest']],'k-')
    plt.plot(x_lim, [seg_hor['waist'],seg_hor['waist']],'k-')
    plt.plot(x_lim, [seg_hor['thigh'],seg_hor['thigh']],'k-')
    plt.plot(x_lim, [seg_hor['knee_upper'],seg_hor['knee_upper']],'k-')
    plt.plot(x_lim, [seg_hor['knee_lower'],seg_hor['knee_lower']],'k-')
    plt.plot(x_lim, [seg_hor['foot'],seg_hor['foot']],'k-');

def find_segmentation_ver_utils_dilate_erode(I):
    import scipy.ndimage as nd
    import copy as cp
    I_change = cp.copy(I)
    struct = np.ones([1, 40])
    I_change = nd.binary_dilation(I_change, struct)
    I_change = nd.binary_erosion(I_change, struct)
    return I_change
    
def find_segmentation_ver_utils_cord_leg(I,  plot_flag = False, ax = None):
    import numpy as np
    n_points = 5;
    n_smooth = 10;
    
    (n_x, n_y) = I.shape
    middle_line = int(n_y /2)

    
    left_position = np.zeros(n_x)
    right_position = np.zeros(n_x)
    for xx in range(n_x):
        this_horizontal_line = I[xx, :]
        line_left = cp.copy(this_horizontal_line)
        line_left[int(n_x/2):] = False 
        if np.sum(line_left) > n_points :
            left_position[xx] =  np.mean(find_matlab_style(line_left, n_points, 'first'))

        line_right = cp.copy(this_horizontal_line)
        line_right [:int(n_x/2)] = False 
        if np.sum(line_right) > n_points :
            right_position[xx]=  np.mean(find_matlab_style(line_right, n_points, 'last'))

                
    left_position = smooth(left_position, n_smooth)
    right_position = smooth(right_position, n_smooth)
    
    left_diff = np.concatenate((np.zeros(1), np.diff(left_position)))
    right_diff = np.concatenate((np.zeros(1), np.diff(right_position)))
    
#     plt.subplot(3,2,1)
#     plt.plot(left_position)
#     plt.subplot(3,2,2)
#     plt.plot(right_position);
#     plt.subplot(3,2,3)
#     plt.plot( np.concatenate((np.zeros(1), np.diff(left_position))));
#     plt.subplot(3,2,4)
#     plt.plot(np.concatenate((np.zeros(1), np.diff(right_position))));
#     plt.show()

    image_mask = np.ones(n_y).astype(np.int64)
    
    dummy, left_foot_top =  find_extreme_in_a_region(left_diff,   image_mask, 'max');
    dummy, right_foot_top =  find_extreme_in_a_region(right_diff, image_mask, 'max');
    dummy, left_foot_bottom =  find_extreme_in_a_region(left_diff,    image_mask, 'min');
    dummy, right_foot_bottom =  find_extreme_in_a_region(right_diff,  image_mask, 'min');
    
    top_line = int((left_foot_top + right_foot_top)/2)
    bottom_line = int((left_foot_bottom + right_foot_bottom)/2)
    
    #     fig = plt.figure(figsize = (4,4))
    
    if plot_flag:
        fig = plt.figure(figsize = (4,4))
        plt.imshow(I)
        print(left_position[left_foot_top + 1])
        print(left_position[left_foot_bottom - 1])
        plt.plot([left_position[left_foot_top + 1], right_position[right_foot_top + 2]], [left_foot_top, right_foot_top],'k-')
        plt.plot([left_position[left_foot_bottom - 1], right_position[right_foot_bottom - 2]], [left_foot_bottom, right_foot_bottom],'r-')
        plt.show()
    
    middle_line = np.array([top_line, bottom_line])
    
    distance_to_center_x =  middle_line - n_x/2
    distance_to_center_y = np.array([0,0]).astype(np.int64)
    return distance_to_center_x, distance_to_center_y

def find_segmentation_ver_utils_cord_trunk(I,  plot_flag = False, ax = None):
    import numpy as np
    n_points = 5;
    n_smooth = 5;
    wid_brain = 30;
    
    (n_x, n_y) = I.shape
    middle_line = int(n_y /2)
    y_cord = np.array([middle_line - wid_brain, middle_line, middle_line + wid_brain])
    n_point_measure = len(y_cord)
    top_position = np.zeros(n_y).astype(np.float64)
    bottom_position = np.zeros(n_y).astype(np.float64)
    for yy in range(n_y):
        this_vertical_line = I[:, yy]
        if np.sum(this_vertical_line) > n_points:
            top_position[yy] = np.mean(find_matlab_style(this_vertical_line, n_points, 'first'))
            bottom_position[yy] = np.mean(find_matlab_style(this_vertical_line, n_points , 'last'))
    left_position = np.zeros(n_x)
    right_position = np.zeros(n_x)
 
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
    
    
    back_cord = top_position[y_cord]
    front_cord = bottom_position[y_cord] 
    
    trunk_x_cord = int((back_cord[1] + front_cord[1])/2)
    left_trunk_y_cord = int(left_position[trunk_x_cord])
    right_trunk_cord =  int(right_position[trunk_x_cord])
    
    # 6 pointd to compute.
#     distance_to_center_top = - np.sqrt((back_cord - middle_line)**2 + (y_cord  - middle_line)**2)
#     distance_to_center_bottom = np.sqrt((front_cord - middle_line)**2 + (y_cord  - middle_line)**2)
    
    if plot_flag:
        if ax is None:
            plt.imshow(I)
            plt.plot(y_cord, back_cord,'k-')
            plt.plot(y_cord, front_cord,'r-')
            plt.plot([left_trunk_y_cord, right_trunk_y_cord], [trunk_x_cord, trunk_x_cord])
            plt.show()

        else:
            ax.imshow(I)
            ax.plot(y_cord, back_cord,'k-')
            ax.plot(y_cord, front_cord,'r-')  
    
    # you should put all four pairs cordinates    
    distance_to_center_x = np.concatenate((back_cord, front_cord, np.array([trunk_x_cord, trunk_x_cord]))) - n_x/2
    distance_to_center_y = np.concatenate((y_cord, y_cord, np.array([left_trunk_y_cord, right_trunk_cord]))) - n_x/2
    
    return distance_to_center_x, distance_to_center_y

def find_segmentation_ver_utils_calculate_cord(I_stack, segmentation_hor, plot_flag = False):
    n_smooth = 10;
    from skimage.filters import sobel
    import copy as cp
    
    (n_x, n_y, n_z) = I_stack.shape
    
    n_lines = 8
    
    ver_cord_x = np.zeros((n_lines, n_z))
    ver_cord_y = np.zeros((n_lines, n_z))
    verline_key = ['back_left','back_middle','back_right','front_left','front_middle','front_right','left_trunk', 'right_trunk']
   

    ## middle line
    groin_to_foot = range(segmentation_hor['groin'], segmentation_hor['foot'])
    distance_to_center_leg_x = np.zeros((2, len(groin_to_foot)))
    distance_to_center_leg_y = np.zeros((2, len(groin_to_foot)))
    for ii, idx in enumerate(groin_to_foot):
        I = np.flipud(I_stack[:,:,idx].transpose())
        I_edge = sobel(I)
        I_mask_this = create_mask_from_image( I_edge)
        I_mask_this = find_segmentation_ver_utils_dilate_erode(I_mask_this)
        distance_to_center_leg_x[:, ii],  distance_to_center_leg_y[:, ii] = find_segmentation_ver_utils_cord_leg(I_mask_this)

    ## middle line + eight four lines.
    chest_to_groin = range(segmentation_hor['chest'], segmentation_hor['groin'])
    distance_to_center_trunk_x = np.zeros((8, len(chest_to_groin)))
    distance_to_center_trunk_y = np.zeros((8, len(chest_to_groin)))

    for ii, idx in enumerate(chest_to_groin):
        I = np.flipud(I_stack[:,:,idx].transpose())
        I_edge =  sobel(I)
        I_mask_this = create_mask_from_image(I_edge)
        I_mask_this = find_segmentation_ver_utils_dilate_erode(I_mask_this)
        distance_to_center_trunk_x[:, ii], distance_to_center_trunk_y[:,ii] = find_segmentation_ver_utils_cord_trunk(I_mask_this)
    
    # get the x and y coordinates.
    ver_cord_x[1,groin_to_foot] = distance_to_center_leg_x[0, :]
    ver_cord_x[4,groin_to_foot] = distance_to_center_leg_x[1, :]    
    ver_cord_x[:,chest_to_groin] = distance_to_center_trunk_x
    
    ver_cord_y[1,groin_to_foot] = distance_to_center_leg_y[0, :]
    ver_cord_y[4,groin_to_foot] = distance_to_center_leg_y[1, :]    
    ver_cord_y[:,chest_to_groin] = distance_to_center_trunk_y
    
    ## smooth signal.
    ver_cord_x_smooth = np.zeros_like(ver_cord_x)
    ver_cord_y_smooth = np.zeros_like(ver_cord_y)
    for ii in range(n_lines):
        ver_cord_x_smooth[ii,:] = smooth(ver_cord_x[ii,:], n_smooth)
        ver_cord_y_smooth[ii,:] = smooth(ver_cord_y[ii,:], n_smooth)
#         
    if plot_flag:
        fig = plt.figure(figsize=(8,8))
        for ii in range(n_lines):
            ax = fig.add_subplot(2, 8, ii + 1)
            ax.plot(ver_cord_x_smooth[ii,:])
        for ii in range(n_lines):
            ax = fig.add_subplot(2, 8, ii + 9)
            ax.plot(ver_cord_y_smooth[ii,:])
        plt.show()
    
    #built a dictionary...
    ver_cord = dict()
    for ii in range(n_lines):
        cord_this_line = np.vstack((ver_cord_x_smooth[ii,:], ver_cord_y_smooth[ii,:]))
        ver_cord[verline_key[ii]] = cord_this_line
    
#    print(ver_cord)
    return ver_cord

def find_segmentation_ver_utils_transform(cord_vect, ang, sign = 1):
    import math
    # 
    cord_dir = np.array([1, math.tan(math.radians(ang + 270))])
    cord_dir = cord_dir/np.linalg.norm(cord_dir)
    
    projected_value = np.dot(cord_dir, cord_vect)
    return projected_value * sign

def find_segmentaion_ver_one_subject(I_aps, ver_cord):
    (n_x, n_y, n_z) = I_aps.shape
    seg_ver = [None for ii in range(n_z)]
    for ang_idx in range(n_z):
        I = np.copy(np.flipud(I_aps[:,:, ang_idx].transpose())) 
        
        ang = (ang_idx + 1) * 360/n_z
        if ang <= 180:
            sign = -1
        else:
            sign = 1
        
        ver_line_dict = dict()
        for key, value in ver_cord.items():
            line = find_segmentation_ver_utils_transform(ver_cord[key], ang, sign) +  n_x/2
            line.astype(np.int64)
            ver_line_dict[key] = line
            
            
        seg_ver[ang_idx] = ver_line_dict  
#    print(seg_ver)
    return seg_ver

def find_segmentation_ver(I_a3d, I_aps, seg_hor):
    I_stack = cp.copy(I_a3d[:, :, -1:0:-1])
#     print(seg_hor)
    ver_cord = find_segmentation_ver_utils_calculate_cord(I_stack, seg_hor, plot_flag = False)
    ver_line =  find_segmentaion_ver_one_subject(I_aps, ver_cord)
    return ver_line

def find_segmentation_ver_utils_plotline(I, seg_ver, ax = None):
    (n_x, n_y) = I.shape
    y_lim = range(n_x - 1);
    if ax is None:
        plt.imshow(I)
        color_string = ['k-','r-','k-','b-','y-','b-','w-','c-']
        counter = 0
        for key, value in seg_ver.items():

            plt.plot(value,y_lim,  color_string[counter])
            counter = counter + 1
    else:
        ax.imshow(I)
        color_string = ['k-','r-','k-','b-','y-','b-','w-','c-']
        counter = 0
        for key, value in seg_ver.items():

            ax.plot(value,y_lim,  color_string[counter])
            counter = counter + 1
