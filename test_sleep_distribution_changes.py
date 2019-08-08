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
df5 = build_df(mydict, time='week5')

# Combine NREM and REM as sleep class
df2['week2', 'SLEEP', 'value'] = df2['week2']['NREM'] + df2['week2']['REM']
df5['week5', 'SLEEP', 'value'] = df5['week5']['NREM'] + df5['week5']['REM']

# Cat both sessions
df = pd.concat([df2, df5], axis=1, sort=True)

def test_changes_between_sessions(mydf, stage):
    d2 = mydf['week2'][stage].dropna()
    d5 = mydf['week5'][stage].dropna()
    d2['name'] = [s.split('_')[0] for s in d2.index.values]
    d5['name']= [s.split('_')[0] for s in d5.index.values]
    matching = pd.merge(d2, d5, on='name') #includy sbjs where sleep stage is present in both sessions
    print(len(matching['value_x']))
    res = stats.wilcoxon(matching['value_x'], matching['value_y'], zero_method='wilcox') #paired testing
    set_trace()
    return res

print('REM {}'.format(test_changes_between_sessions(df, stage='REM')))
#print('NREM {}'.format(test_changes_between_sessions(df, stage='NREM')))
#print('WAKE {}'.format(test_changes_between_sessions(df, stage='WAKE')))
#print('SLEEP {}'.format(test_changes_between_sessions(df, stage='SLEEP')))
