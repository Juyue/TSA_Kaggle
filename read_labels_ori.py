import pandas as pd
import numpy as np

def read_labels_ori(label_file_fullpath):

    n_zone_max = 17
    unsorted_df = pd.read_csv(label_file_fullpath)
    
    # get subject id
    s = list(range(0,len(unsorted_df),n_zone_max ))
    obs = unsorted_df.loc[s,'Id'].str.split('_')
    scan_ID = [x[0] for x in obs]
    
    
    columns = sorted(['zone' + str(i + 1) for i in range(n_zone_max)])
    df = pd.DataFrame(index = scan_ID, columns = columns)

    for i in range(n_zone_max):
        s = range(i, len(unsorted_df), n_zone_max)
        df.iloc[:,i] = unsorted_df.iloc[s, 1].values
       
    return df
