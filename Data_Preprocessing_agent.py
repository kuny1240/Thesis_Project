import pandas as pd
import numpy as np
import os

path = '.\Raw_Data'

list_dir = os.listdir(path)

frames = []

actions = ['CELLMLB_INTERFREQMLBUENUMTHD',
            'CELLMLB_UENUMDIFFTHD',
            'CELLMLB_CELLCAPACITYSCALEFACTOR',
            'INTERFREQHOGROUP_INTERFREQHOA4THDRSRP']

states = pd.read_csv('./Configures/States_Names_1.csv')['Useful_Config'].values

states = list(np.reshape(states
                         ,(31,)))

Identifies = ['CELL_CELLNAME','TIME']

bad_cells = pd.read_csv('./Configures/cell_bad.csv',encoding='gb2312').values

label = ['L_USER_DL_THROUGPUT_FULLBUFF_MBPS']

Useful_cols = Identifies + actions + states + label

for d in list_dir:
    tmp_d = os.path.join(path, d)
    tmp_df = pd.read_csv(tmp_d, encoding='gb2312',usecols=Useful_cols)
    frames.append(tmp_df)

raw_data = pd.concat(frames, sort=False)

sort_data = raw_data.sort_values(['CELL_CELLNAME', 'TIME'], ascending=[True, True])

sort_data['TIME'] = pd.to_datetime(sort_data['TIME'])

cells = sort_data['CELL_CELLNAME'].unique()#SECTOR_IDENTIFICATION

i = 0

frames = []

for c in cells:
    if c in bad_cells:
        print(c)
        continue
    i += 1
    df = sort_data.loc[sort_data['CELL_CELLNAME'] == c] #Temperary dataframe saving the sector
    df = df.resample('D',on = 'TIME').mean()
    df['CELL_CELLNAME'] = c
    df.dropna(axis = 1)
    df.reset_index()
    df.fillna(method='bfill',inplace=True)
    df.dropna(axis = 0,inplace = True)
    frames.append(df)

result = pd.concat(frames)

result.to_csv('./Processed_Data/Basic_agent_data_v1.csv')





