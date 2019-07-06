import pandas as pd
import numpy as np
import os


path = '.\Raw_Data'

out_path = ".\Pruned_Data"

if not os.path.isdir(out_path):
    os.makedirs(out_path)


list_dir = os.listdir(path)


def drop_nonuseful_cols(raw_data, threshold=0.8):
    ''' This is a function designed for this purpose: dropping all the
    columns that are not useful for the system, what is not useful?
    the columns are missing too many points
    Input: raw_data, the original data we take as an input, DF;
    threshold, how we define a column as useless, float
    '''

    count = raw_data.count()
    cols = list(raw_data.columns)
    max_row_num = raw_data.shape[0]
    threshold_row_num = max_row_num * threshold

    for col in cols:
        if count[col] < threshold_row_num:
            raw_data.drop(col, axis=1, inplace=True)

    return raw_data

for d in list_dir:
    tmp_dir = os.path.join(path, d)
    tmp_df = pd.read_csv(tmp_dir, encoding='gb2312')
    print("Reading "+ tmp_dir + " Successful, data shape is: " , tmp_df.shape)
    # Se
    tmp_df = tmp_df.select_dtypes(include=[np.number])
    print("Eliminating None numeric data successful! Current data shape:" , tmp_df.shape)
    tmp_df = drop_nonuseful_cols(tmp_df)
    print("Drop none successful data successful! Current data shape:", tmp_df.shape)
    tmp_df = tmp_df.dropna(axis=0)
    print("Drop vaccant column successful! Current data shape:",tmp_df.shape)

    tmp_df.to_csv(out_path+"-pruned-"+d)