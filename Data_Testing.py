'''
This file choose the columns that will be used training for the simulator from the Raw_Data then
save them to as processed data, this part is simple.
'''
import pandas as pd
import numpy as np
import os

# set up for the columns
path = '.\Test_Data'

list_dir = os.listdir(path)

frames = []

actions1 = ['CELLDLSCHALGO_RBPRIMCSSELECTRATIOTHD',
           'CELLCQIADJALGO_INITDELTACQI',
           'CELLCQIADJALGO_CQIADJSTEP',
           'CELLCQIADJALGO_INITDLIBLERTARGET',
           'CELLPCALGO_PUSCHRSRPHIGHTHD']

actions2 = ['CELLMLB_INTERFREQMLBUENUMTHD',
            'CELLMLB_UENUMDIFFTHD',
            'CELLMLB_CELLCAPACITYSCALEFACTOR']

label1 = ['INTERFREQHOGROUP_INTERFREQLOADBASEDHOA4THDRSRP','INTERFREQHOGROUP_INTERFREQHOA4THDRSRP']

label2 = ['INTERFREQHOGROUP_INTERFREQHOA1THDRSRP','INTERFREQHOGROUP_INTERFREQHOA2THDRSRP']

Useful_cols = actions1+actions2 + label1 + label2

#Read the useful columns from rawdata
for d in list_dir:
    tmp_d = os.path.join(path, d)
    tmp_df = pd.read_csv(tmp_d, encoding='gb2312',usecols = Useful_cols)
    frames.append(tmp_df)#read all data in the raw data file

raw_data = pd.concat(frames, sort=False)

raw_data = raw_data.select_dtypes(exclude=['object'])

raw_data.fillna(raw_data.mean(),inplace = True)

raw_data.dropna(axis = 1,inplace = True)

# raw_data.drop(raw_data[raw_data.L_USER_DL_THROUGPUT_FULLBUFF_MBPS == 0].index,inplace = True)

raw_data.to_csv('./Test_Data/test_data.csv')


