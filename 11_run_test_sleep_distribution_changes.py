'''
Test statistically whether duration of sleep stages differ across sessions
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2, mysave
import matplotlib.pyplot as plt
import os
import seaborn as sns
from collections import Counter
from scipy import stats

matplotlib.rcParams.update({'font.size': 12,'xtick.labelsize':8, 'ytick.labelsize':8})
np.random.seed(12)

path = 'H:\\BABY\\working\\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

events_id = ['N', 'R', 'W']

mydict = {'week2': {'NREM': {'value':[], 'sbj':[]},
                    'REM': {'value':[], 'sbj':[]},
                    'WAKE':{'value':[], 'sbj':[]}},
        'week5': {'NREM': {'value':[], 'sbj':[]},
                'REM': {'value':[], 'sbj':[]},
                'WAKE':{'value':[], 'sbj':[]}} }

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        print(sbj)
        if len(os.listdir(os.path.dirname(paths(typ='epoch', sbj=sbj)))) == 0: #emty folder means bad sbj
            continue
        else:
            epoch = myload(typ='epoch', sbj=sbj)
            epoch = epoch[events_id]
            events = epoch.events[:,-1]
            del epoch
            count = Counter(events)
            for s, idx in zip(['NREM', 'REM', 'WAKE'], [1,2,3]):
                if idx in list(count.keys()):
                    #set_trace()
                    mydict[time][s]['value'].append(count[idx])
                    mydict[time][s]['sbj'].append(sbj)
                else:
                    mydict[time][s]['value'].append(np.nan)
                    mydict[time][s]['sbj'].append(np.nan)

def build_df(mydict, time='week2'):
    '''
    Build pandas data frame from nested dictionary.
    '''
    mydf = pd.DataFrame.from_dict({(i,j,k): mydict[i][j][k]
                                for i in [time]
                                for j in mydict[i].keys()
                                for k in mydict[i][j].keys()},
                                orient='columns')

    sbj_columns = [l for l in mydf.columns.tolist() if 'sbj' in l]
    sbjs = mydf[sbj_columns]
    mydf = mydf.drop(columns=sbj_columns)
    sbjs['new_idx'] = sbjs[time].apply(lambda x: ' '.join(x.dropna().astype(str).unique()), axis=1)
    mydf = mydf.set_index(sbjs['new_idx'])
    return mydf

# Convert nested dict  to pandas df
df2 = build_df(mydict, time='week2')
df2 = df2.fillna(0)
df5 = build_df(mydict, time='week5')
df5 = df5.fillna(0)

#Drop mulitindexes columns
mi = df2.columns.droplevel((0,2))
df2.columns = mi
df5.columns = mi
df2['time'] = ['week2'] * len(df2)
df5['time'] = ['week5'] * len(df5)

# Cat both sessions
df = pd.concat([df2, df5], axis=0, sort=True)

# Combine NREM and REM as sleep class
df['SLEEP'] = df['NREM'] + df['REM']

# Total
df['TOTAL'] = df['NREM'] + df['REM'] +  df['WAKE']

# Print ratios
df['NREM_ratio'] = df['NREM'] / df['TOTAL']
df['REM_ratio'] = df['REM'] / df['TOTAL']
df['WAKE_ratio'] = df['WAKE'] / df['TOTAL']
df = df.round(2)
df[['time', 'NREM_ratio']].groupby('time').mean()
df[['time', 'REM_ratio']].groupby('time').mean()
df[['time', 'WAKE_ratio']].groupby('time').mean()


def test_changes_between_sessions(mydf, stage):
    d2 = mydf[[stage, 'time']][mydf['time'] == 'week2']
    d5 = mydf[[stage, 'time']][mydf['time'] == 'week5']
    d2['name'] = [s.split('_')[0] for s in d2.index.values]
    d5['name']= [s.split('_')[0] for s in d5.index.values]
    d2 = d2.replace(0, np.nan).dropna()
    d5 = d5.replace(0, np.nan).dropna()
    matching = pd.merge(d2, d5, on='name') #includy sbjs where sleep stage is present in both sessions
    col1 = '{}_x'.format(stage)
    col2 = '{}_y'.format(stage)
    res = stats.wilcoxon(matching[col1], matching[col2], zero_method='wilcox') #paired testing
    #set_trace()
    medians = [np.median(matching[col1]), np.median(matching[col2])]
    return res, medians

print('REM: stat{}, medians w2 vs w5{}'.format(*test_changes_between_sessions(df, stage='REM')))

print('NREM {}'.format(test_changes_between_sessions(df, stage='NREM')))
print('WAKE {}'.format(test_changes_between_sessions(df, stage='WAKE')))
print('SLEEP {}'.format(test_changes_between_sessions(df, stage='SLEEP')))
