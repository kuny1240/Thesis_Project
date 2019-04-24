import pandas as pd
import numpy as np
import os

path = '.\Raw_Data'

list_dir = os.listdir(path)

frames = []

actions = ['CELLDLSCHALGO_RBPRIMCSSELECTRATIOTHD',
           'CELLCQIADJALGO_INITDELTACQI',
           'CELLCQIADJALGO_CQIADJSTEP',
           'CELLCQIADJALGO_INITDLIBLERTARGET',
           'CELLPCALGO_PUSCHRSRPHIGHTHD']

states = pd.read_csv('./Configures/States_Names.csv').values

states = list(np.reshape(states
                         ,(151,)))

Identifies = ['CELL_CELLNAME','TIME']

#label = ['L_THRP_DL_BITRATE_LS5M_RATIO']

label = ['L_USER_DL_THROUGPUT_FULLBUFF_MBPS']

Useful_cols = Identifies + actions + states + label

for d in list_dir:
    tmp_d = os.path.join(path, d)
    tmp_df = pd.read_csv(tmp_d, encoding='gb2312',usecols=Useful_cols)
    frames.append(tmp_df)

raw_data = pd.concat(frames, sort=False)

sort_data = raw_data.sort_values(['CELL_CELLNAME', 'TIME'], ascending=[True, True])

sort_data['TIME'] = pd.to_datetime(sort_data['TIME'])

sectors = sort_data['CELL_CELLNAME'].unique()#SECTOR_IDENTIFICATION

i = 0

frames = []

for s in sectors:
    i += 1
    df = sort_data.loc[sort_data['CELL_CELLNAME'] == s] #Temperary dataframe saving the sector
    df = df.resample('D',on = 'TIME').mean()
    df['Episode'] = [i] * df.shape[0]
    df['CELL_CELLNAME'] = s
    df.dropna(axis = 1)
    df.reset_index()
    print(df.loc[:,actions+label])
    frames.append(df)

result = pd.concat(frames)

result.to_csv('./Processed_Data/Basic_agent_data.csv')





