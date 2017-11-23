from find_segmentation_line import * 

def zone_segmentation_utils_hor_to_zone():
    n_zone = 17     
    hor_start = np.zeros(n_zone)
    hor_end = np.zeros(n_zone)
    
    ver_start = np.array([2,0,3,0,4, 5,5,6,6,6,7, 7, 11,11,12,12,4]).astype(np.int64)
    ver_end = np.array(  [5,2,5,3,5, 6,6,7,7,7,12,12,13,13,10,10,5]).astype(np.int64)
    
    seg_hor_zone = {'start': ver_start,
                    'end': ver_end}
    return  seg_hor_zone
    
    
def zone_segmentation_utils_ver_to_zone():
    n_zone = 17
    n_image = 64 

    ver_start = np.zeros((n_zone, n_image))
    ver_end = np.zeros((n_zone, n_image))
    flag = np.zeros((n_zone, n_image)).astype(np.bool)
    
    image_id_1_1 = np.arange(0, 9)
    image_id_1_2 =  np.arange(26, 41)
    image_id_1_3 = np.arange(52,64)
    image_id_1 = np.concatenate((image_id_1_1 , image_id_1_2, image_id_1_3))
    flag[0, image_id_1] = True
    ver_start[0, image_id_1_1] = 0
    ver_end[0,image_id_1_1] = 7
    ver_start[0, image_id_1_2] = 7
    ver_end[0,image_id_1_2] = 9
    ver_start[0, image_id_1_3] = 0
    ver_end[0,image_id_1_3] = 7
    
    
    image_id_2_1 = np.arange(0, 13)
    image_id_2_2 =  np.arange(18, 45)
    image_id_2_3 = np.arange(49,64)
    image_id_2 = np.concatenate((image_id_2_1 , image_id_2_2, image_id_2_3))
    flag[1, image_id_2] = True
    ver_start[1, image_id_2_1] = 0
    ver_end[1,image_id_2_1] = 5
    ver_start[1, image_id_2_2] = 5
    ver_end[1,image_id_2_2] = 9
    ver_start[1, image_id_2_3] = 0
    ver_end[1,image_id_2_3] = 5
    
    image_id_3_1 = np.arange(0, 14)
    image_id_3_2 =  np.arange(22, 39)
    image_id_3_3 = np.arange(55,64)
    image_id_3 = np.concatenate((image_id_3_1 , image_id_3_2, image_id_3_3))
    flag[2, image_id_3] = True
    ver_start[2, image_id_3_1] = 8
    ver_end[2,image_id_3_1] = 9
    ver_start[2, image_id_3_2] = 0
    ver_end[2,image_id_3_2] = 8
    ver_start[2, image_id_3_3] = 8
    ver_end[2,image_id_3_3] = 9
    
    image_id_4_1 = np.arange(0, 13)
    image_id_4_2 =  np.arange(18, 43)
    image_id_4_3 = np.arange(52,64)
    image_id_4 = np.concatenate((image_id_4_1 , image_id_4_2, image_id_4_3))
    flag[3, image_id_4] = True
    ver_start[3, image_id_4_1] = 5
    ver_end[3,image_id_4_1] = 9
    ver_start[3, image_id_4_2] = 0
    ver_end[3,image_id_4_2] = 5
    ver_start[3, image_id_4_3] = 5
    ver_end[3,image_id_4_3] = 9
    
    image_id_5_1 = np.arange(0, 8)
    image_id_5_2 =  np.arange(56, 64)
    image_id_5 = np.concatenate((image_id_5_1 , image_id_5_2))
    flag[4, image_id_5] = True
    ver_start[4, image_id_5_1] = 7
    ver_end[4,image_id_5_1] = 8
    ver_start[4, image_id_5_2] = 7
    ver_end[4,image_id_5_2] = 8

    image_id_17_1 = np.arange(24, 39)
    image_id_17 = image_id_17_1
    flag[16, image_id_17] = True
    ver_start[16, image_id_17_1] = 8
    ver_end[16,image_id_17_1] = 7

    image_id_6_1 = np.arange(0, 8)
    image_id_6_2 =  np.arange(20, 48)
    image_id_6_3 = np.arange(48,64)
    image_id_6 = np.concatenate((image_id_6_1 , image_id_6_2, image_id_6_3))
    flag[5, image_id_6] = True
    ver_start[5, image_id_6_1] = 0
    ver_end[5,image_id_6_1] = 5
    ver_start[5, image_id_6_2] = 2
    ver_end[5,image_id_6_2] = 9
    ver_start[5, image_id_6_3] = 0
    ver_end[5,image_id_6_3] = 5
    
    # need to be checked later on.
    image_id_7_1 = np.arange(0, 16)
    image_id_7_2 =  np.arange(16, 32)
    image_id_7_3 = np.arange(32,40)
    image_id_7_4 = np.arange(55,64)
    image_id_7 = np.concatenate((image_id_7_1 , image_id_7_2, image_id_7_3, image_id_7_4))
    flag[6, image_id_7] = True
    ver_start[6, image_id_7_1] = 5
    ver_end[6,image_id_7_1] = 9
    ver_start[6, image_id_7_2] = 0
    ver_end[6,image_id_7_2] = 2
    ver_start[6, image_id_7_3] = 0
    ver_end[6,image_id_7_3] = 2
    ver_start[6, image_id_7_4] = 5
    ver_end[6,image_id_7_4] = 9
    
    image_id_8_1 = np.arange(0, 7)
    image_id_8_2 =  np.arange(24, 48)
    image_id_8_3 = np.arange(48,64)
    image_id_8 = np.concatenate((image_id_8_1 , image_id_8_2, image_id_8_3))
    flag[7, image_id_8] = True
    ver_start[7, image_id_8_1] = 0
    ver_end[7,image_id_8_1] = 4
    ver_start[7, image_id_8_2] = 1
    ver_end[7,image_id_8_2] = 9
    ver_start[7, image_id_8_3] = 0
    ver_end[7,image_id_8_3] = 4
    
    image_id_9_1 = np.arange(0, 8)
    image_id_9_2 =  np.arange(18, 40)
    image_id_9_3 = np.arange(54,64)
    image_id_9 = np.concatenate((image_id_9_1 , image_id_9_2, image_id_9_3))
    flag[8, image_id_9] = True
    ver_start[8, image_id_9_1] = 4
    ver_end[8,image_id_9_1] = 6
    ver_start[8, image_id_9_2] = 3
    ver_end[8,image_id_9_2] = 1
    ver_start[8, image_id_9_3] = 4
    ver_end[8,image_id_9_3] = 6
    
    image_id_10_1 = np.arange(0, 16)
    image_id_10_2 =  np.arange(16, 38)
    image_id_10_3 = np.arange(58,64)
    image_id_10 = np.concatenate((image_id_10_1 , image_id_10_2, image_id_10_3))
    flag[9, image_id_10] = True
    ver_start[9, image_id_10_1] = 6
    ver_end[9,image_id_10_1] = 9
    ver_start[9, image_id_10_2] = 0
    ver_end[9,image_id_10_2] = 3
    ver_start[9, image_id_10_3] = 6
    ver_end[9,image_id_10_3] = 9
    
    image_id_11_1 = np.arange(0, 11)
    image_id_11_2 =  np.arange(17, 48)
    image_id_11_3 = np.arange(48,64)
    image_id_11 = np.concatenate((image_id_11_1 , image_id_11_2, image_id_11_3))
    flag[10, image_id_11] = True
    ver_start[10, image_id_11_1] = 0
    ver_end[10,image_id_11_1] = 5
    ver_start[10, image_id_11_2] = 2
    ver_end[10,image_id_11_2] = 9
    ver_start[10, image_id_11_3] = 0
    ver_end[10,image_id_11_3] = 5
    
    # section 13
    flag[12, image_id_11] = True
    ver_start[12, :] = ver_start[10, :]
    ver_end[12,:] = ver_end[10,:]
    
    # secontion 15
    flag[14, image_id_11] = True
    ver_start[14, :] = ver_start[10, :]
    ver_end[14,:] = ver_end[10,:]
    
    
    image_id_12_1 = np.arange(0, 16)
    image_id_12_2 =  np.arange(16, 45)
    image_id_12_3 = np.arange(52,64)
    image_id_12 = np.concatenate((image_id_12_1 , image_id_12_2, image_id_12_3))
    flag[11, image_id_12] = True
    ver_start[11, image_id_12_1] = 5
    ver_end[11,image_id_12_1] = 9
    ver_start[11, image_id_12_2] = 0
    ver_end[11,image_id_12_2] = 2
    ver_start[11, image_id_12_3] = 5
    ver_end[11,image_id_12_3] = 9
    
    # section 14. should follow 12
    flag[13, image_id_12] = True
    ver_start[13, :] = ver_start[11, :]
    ver_end[13,:] = ver_end[11,:]
   
    # section 16. should follow 12
    flag[15, image_id_12] = True
    ver_start[15, :] = ver_start[11, :]
    ver_end[15,:] = ver_end[11,:]
    
    ver_start  = ver_start.astype(np.int64)
    ver_end = ver_end.astype(np.int64)
    
    seg_ver_zone = {'start': ver_start,
                    'end':   ver_end,
                    'flag': flag}
    return seg_ver_zone

def zone_segmentation_utils_verline_number_to_name():
    num_to_key = {0:'0',
            1: 'back_left',
            2: 'back_middle',
            3: 'back_right',
            4: 'front_left',
            5: 'front_middle',
            6: 'front_right',
            7: 'left_trunk',
            8: 'right_trunk',
            9: 'ny'}
    return num_to_key 
def zone_segmentation_utils_horline_number_to_name():
    num_to_key = {0:'0',
            1: 'head',
            2: 'left_elbow',
            3: 'right_elbow',
            4: 'neck',
            5: 'chest',
            6: 'waist',
            7: 'thigh',
            8: 'knee_upper',
            9: 'knee_lower',
            10: 'nx',
            11: 'lower_thigh',
            12: 'knee',
            13: 'middle_calf'}
    return num_to_key 


def zone_segmentation_utils_fill_in_one_zone(zone_mask, zone_num, seg_hor_start, seg_hor_end, seg_ver_start, seg_ver_end):
    # seg_hor_start, seg_hor_end will be two scalers
    # seg_ver_start, seg_ver_end will be two vectors.
    if len(seg_ver_start) == 1:
        zone_mask[seg_hor_start: seg_hor_end,  seg_ver_start: seg_ver_end, zone_num]= True
    else:
        for ii in range(seg_hor_start, seg_hor_end):
            zone_mask[ii, seg_ver_start[ii]: seg_ver_end[ii], zone_num] = True

    
def image_segmentation_line_to_zone(seg_hor, seg_ver, I_mask, seg_ver_zone, seg_hor_zone, image_num):
    n_zone = 17;
    (n_x, n_y) = I_mask.shape
    middle_of_elbow = int((seg_hor['left_elbow'] + seg_hor['right_elbow'])/2);
    middle_y = int(n_y/2)
    
    zone_mask = np.zeros((n_x, n_y, n_zone)) == 1
    ver_0 = np.zeros(n_x).astype(np.int64)
    ver_ny = np.ones(n_x).astype(np.int64) * (n_y - 1)
    
    ## you have to make verything into a big dictionary...
    ## for the left_trunk and right trunk. you will have to change it.
    seg_ver_used = cp.copy(seg_ver)
    left_trunk  = np.mean(seg_ver['left_trunk' ][seg_hor['chest']:seg_hor['waist']]) * np.ones(n_y)
    right_trunk = np.mean(seg_ver['right_trunk'][seg_hor['chest']:seg_hor['waist']]) * np.ones(n_y)
     
    seg_ver_used['left_trunk'] = left_trunk.astype(np.int64)
    seg_ver_used['right_trunk'] = right_trunk.astype(np.int64)
    seg_ver_used['0'] = np.zeros(n_x).astype(np.int64)
    seg_ver_used['ny'] = np.ones(n_x).astype(np.int64) * (n_y - 1)
   
    seg_hor_used = cp.copy(seg_hor)
    seg_hor_used['0'] = 0
    seg_hor_used['nx'] = n_x - 1
    
    # turn seg_hor, seg_ver into integers.
    for key, value in seg_ver_used.items():
        seg_ver_used[key] = seg_ver_used[key].astype(np.int64)
    
    flag = seg_ver_zone['flag'][:, image_num]
    ver_start = seg_ver_zone['start'][:, image_num]
    ver_end   = seg_ver_zone['end'][:, image_num]
    hor_start = seg_hor_zone['start']
    hor_end = seg_hor_zone['end']
          
    ver_key = zone_segmentation_utils_verline_number_to_name()
    hor_key = zone_segmentation_utils_horline_number_to_name()
    for ii in range(n_zone):
        if flag[ii]:
#             print('region %d is being calculated' % (ii))
#             print(zone_mask.shape)
            zone_segmentation_utils_fill_in_one_zone(zone_mask, ii, seg_hor_used[hor_key[hor_start[ii]]], seg_hor_used[hor_key[hor_end[ii]]],
                                                     seg_ver_used[ver_key[ver_start[ii]]], seg_ver_used[ver_key[ver_end[ii]]])
    # do not use the zone_mask combine, because there is overlapping. 
    
#    zone_mask_combine = np.zeros((n_x, n_y));
#    for zz in range(n_zone):
#        zone_mask_combine[zone_mask[:,:,zz] & I_mask] = zz + 1;
    
    return zone_mask

def image_segmentation_plot_utils_find_range(mask):
    ver_start_list = []
    ver_end_list = []
    for ii in range(mask.shape[1]):
        ver_start_list = ver_start_list + find_matlab_style(mask[:,ii], n = 1, mode = 'first')
        ver_end_list = ver_end_list + find_matlab_style(mask[:,ii], n = 1, mode = 'last')
    ver_start = np.min(ver_start_list)
    ver_end = np.max(ver_end_list)

    hor_start_list = []
    hor_end_list = []
    for ii in range(mask.shape[0]):
        hor_start_list = hor_start_list + find_matlab_style(mask[ii,:], n = 1, mode = 'first')
        hor_end_list = hor_end_list + find_matlab_style(mask[ii,:], n = 1, mode = 'last')
    hor_start = np.min(hor_start_list)
    hor_end = np.max(hor_end_list)
    
    return ver_start, ver_end, hor_start, hor_end 


def image_segmentation_plot_utils_find_center_of_mass(mask):
    ver_start, ver_end, hor_start, hor_end = image_segmentation_plot_utils_find_range(mask)
    center_of_mass = np.array([(ver_start + ver_end)/2, (hor_start+ hor_end)/2])
    
    return center_of_mass

def image_segmentation_plot_zone_use_zone_mask(zone_mask,ax = None):
    # define 17 colors for body zone,
    n_x, n_y, n_zone = zone_mask.shape
    zone_mask_combine = np.zeros((n_x, n_y));
    for zz in range(n_zone):
        zone_mask_combine[zone_mask[:,:,zz]] = zz + 1;
    if ax is None:
        fig, ax = plt.subplots()
    imgplot = ax.imshow(zone_mask_combine, clim = (0, 18), cmap = 'tab20')
#     ax.colorbar(ticks = range(18))
    image_segmentation_plot_utils_write_zone_number_use_zone_mask(zone_mask, ax)

def image_segmentation_plot_utils_write_zone_number_use_zone_mask(zone_mask, ax):
    for ii in range(17):
        mask_this = zone_mask[:, :, ii]
        if np.sum(mask_this.flatten()) > 0:
            center_of_mass = image_segmentation_plot_utils_find_center_of_mass(mask_this)
            ax.text(center_of_mass[1], center_of_mass[0], str(ii),fontsize = 15)
def image_segmentation_plot_zone(zone_mask_combine,ax = None):
    # define 17 colors for body zone,
    if ax is None:
        fig, ax = plt.subplots()
    imgplot = ax.imshow(zone_mask_combine, clim = (0, 18), cmap = 'tab20')
#     ax.colorbar(ticks = range(18))
    image_segmentation_plot_utils_write_zone_number(zone_mask_combine, ax)
    
    
def image_segmentation_plot_utils_write_zone_number(zone_mask_combine, ax):
    for ii in range(1, 18):
        mask_this = zone_mask_combine == ii
        if np.sum(mask_this.flatten()) > 0:
            center_of_mass = image_segmentation_plot_utils_find_center_of_mass(mask_this)
            ax.text(center_of_mass[1], center_of_mass[0], str(ii),fontsize = 15)

    
def find_segmentation_zone_main(data):
    I_ver_orig = data['.a3d']
    I_front = np.flipud(data['.a3daps'][:,:,0].transpose())
    I_front_mask = create_mask_from_image(I_front)
    seg_hor = find_segmentation_hor(I_front_mask, plot_flag = False)
    seg_ver = find_segmentation_ver(data['.a3d'], data['.a3daps'], seg_hor)
    
    seg_ver_zone = zone_segmentation_utils_ver_to_zone()
    seg_hor_zone = zone_segmentation_utils_hor_to_zone()
    
    zone_mask_list = []
    for ii in range(data['.a3daps'].shape[2]):
        I = np.flipud(data['.a3daps'][:,:,ii].transpose())
        I_mask = np.ones_like(I) == 1
        ## Use full_I_mask, maybe do some cut??
        ## use contour instead of I_mask...
        zone_mask_combine = image_segmentation_line_to_zone(seg_hor, seg_ver[ii],  I_mask, seg_ver_zone, seg_hor_zone, ii)

        zone_mask_list.append(zone_mask_combine)
    
    return zone_mask_list
    