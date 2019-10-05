from functional import read_pickle
import pandas as pd
import numpy as np 
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

psd = read_pickle('psd_for_stages_times_F3F4.txt')
freqs_bins = [10, 11, 12,13,14]
stages = ['NREM', 'REM', 'WAKE']
times = ['week2', 'week5']
#items = ['psd', 'name']
store_psd = {t: {s: [] for s in stages} for t in times } 
store_names = {t: {s: [] for s in stages} for t in times } 
mspescale = 1 

for t in times:
    for s in stages:
        d = psd[s][t][0]
        names = psd[s][t][1]
        d = d[np.logical_not(np.isnan(d))].reshape([-1, 29]) # no freqs bins = 29
        d = d[:, freqs_bins] #select bins
        d = d.mean(1) # av bins    
        store_psd[t][s] = d 
        store_names[t][s]  = names

# Nested dict to df for psd
df_psd = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in store_psd.items()
        },
        axis=0)    
df_psd = df_psd.reset_index().rename(columns={'level_0' : 'time', 'level_1' : 'stag'})
df_psd_l = pd.melt(df_psd, id_vars=['time', 'stag'], value_name='psd')

# Nested dict to df for names
df_names = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in store_names.items()
        },
        axis=0)    
df_names = df_names.reset_index().rename(columns={'level_0' : 'time', 'level_1' : 'stag'})
df_names_l = pd.melt(df_names, id_vars=['time', 'stag'], value_name='name')

# Merge psd with names
dfm = df_psd_l.merge(df_names_l)

#Load mspe 
df_mspe = pd.read_csv('H:\\BABY\\results\\figs\\mspe_final\\factorchannels_mspe_scale{}.csv'.format(mspescale))
df_mspe = df_mspe[df_mspe['channels'] == 'front'] #PSD for C3-C4 
df_mspe = df_mspe.rename(columns={'new_idx' : 'name'})

# merge psd with mspe
dfm = dfm.merge(df_mspe, on=['name', 'stag', 'time'])


def spearmanr(x, y):
        return stats.spearmanr(x, y)

nrem = dfm[dfm['stag'] == 'NREM']
rem = dfm[dfm['stag'] == 'REM']
wake = dfm[dfm['stag'] == 'WAKE']
#sns.set(rc={'figure.figsize':(6, 6)})
sns.set_style('ticks')
matplotlib.rcParams.update({'font.size': 17,'xtick.labelsize':14, 'ytick.labelsize':14, 'axes.labelsize': 'medium', 'figure.figsize' : (15, 15)})

'''
for d, c, n in zip([nrem, rem, wake], ['royalblue', 'indianred', 'tan'], ['nrem', 'rem', 'wake']):     
        sns.jointplot('psd', 'value', data=d, kind='reg', color=c, stat_func=spearmanr)
        plt.xlabel('Sigma power [dB]')
        plt.ylabel('Entropy (scale={}) [bit]'.format(mspescale))
        #plt.ylim((1.2, 1.5))
        #wplt.show()
        dpi = 300
        plt.savefig('corr_{}_mspescale{}.tiff'.format(n, mspescale), dpi=dpi)
        #break
'''

# Drop not needed cols 
dfm = dfm[['time', 'stag', 'psd', 'value', 'name_id_short']]
# Reshape
dfmp = dfm.pivot_table(index=['time','name_id_short'], columns='stag').reset_index()
#Mark all duplicates as True, mask subjects occuring only once
mask = pd.Series(dfmp['name_id_short']).duplicated(keep=False).values
dfmp = dfmp.iloc[mask, :]

# Subtract sessions
w2 = dfmp[dfmp['time']=='week2'][['value', 'psd']].reset_index(drop=True)
w2 = (w2-w2.mean(skipna=True))/w2.std(skipna=True)

n2 = dfmp[dfmp['time']=='week2']['name_id_short'].values
w5 = dfmp[dfmp['time']=='week5'][['value', 'psd']].reset_index(drop=True)
w5 = (w5-w5.mean(skipna=True))/w5.std(skipna=True)

n5 = dfmp[dfmp['time']=='week5']['name_id_short'].values
assert all(n2==n5)
diff = w5.subtract(w2)
diff['names'] = n2

 
psd_melt = diff['psd'].melt(value_name='psd')
mspe_melt = diff['value'].melt(value_name='mspe')

diffm = pd.concat([psd_melt, mspe_melt.drop(columns='stag')], 1)

sns.lmplot(data=diffm, x='psd', y='mspe', hue='stag')


#sns.jointplot(data=diffm[diffm['stag']=='NREM'], x='psd', y='mspe')
plt.show()

from sklearn.preprocessing import StandardScaler