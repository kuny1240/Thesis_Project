'''
This file choose the columns that will be used training for the simulator from the Raw_Data then
save them to as processed data, this part is simple.
'''
import pandas as pd
import numpy as np
import os

# set up for the columns
path = '.\Raw_Data'

list_dir = os.listdir(path)

frames = []

actions = ['CELLDLSCHALGO_RBPRIMCSSELECTRATIOTHD',
           'CELLCQIADJALGO_INITDELTACQI',
           'CELLCQIADJALGO_CQIADJSTEP',
           'CELLCQIADJALGO_INITDLIBLERTARGET',
           'CELLPCALGO_PUSCHRSRPHIGHTHD']

states = pd.read_csv('./Configures/States_Names.csv').values

states = list(np.reshape(states,(151,)))

label = ['L_THRP_DL_BITRATE_LS5M_RATIO']

Useful_cols = actions + states + label

#Read the useful columns from rawdata
for d in list_dir:
    tmp_d = os.path.join(path, d)
    tmp_df = pd.read_csv(tmp_d, encoding='gb2312',usecols=Useful_cols)
    frames.append(tmp_df)#read all data in the raw data file

raw_data = pd.concat(frames, sort=False)

raw_data.fillna(raw_data.mean(),inplace = True)

raw_data.dropna(axis = 1,inplace = True)

raw_data.drop(raw_data[raw_data.L_THRP_DL_BITRATE_LS5M_RATIO == 0].index,inplace = True)

raw_data.to_csv('./Processed_Data/simulator_data.csv')


