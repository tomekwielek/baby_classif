import os
import mne
from config import raw_path, myload, base_path
from mne import io
from matplotlib import pyplot as plt
from functional import (load_single_append, select_stages_to_classif, merge_stages,
                         count_stag)
import pandas as pd
import numpy as np
import seaborn as sns
fnames =  os.listdir(base_path)

fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')]
mapper = {'N' : 1, 'R' : 1, 'W' : 3} #for stages merging
sel_idxs = [1, 2, 3]
merge = False

def get_data_wrap(fnames, merge=False, setup = 'pet1m3'):
    pe, stag, sbj_names = load_single_append(base_path, fnames, typ = setup)
    pe, stag = select_stages_to_classif(pe, stag, sel_idxs=sel_idxs)
    if merge:
        stag = merge_stages(stag, mapper) #merge stages; typicaly N and R
    return pe, stag, sbj_names

pe1, stag1, sbj_names1 = get_data_wrap(fnames1, setup = 'pet1m3')
pe2, stag2, sbj_names2 = get_data_wrap(fnames2, setup = 'pet1m3')

pe1_uncor, stag1_uncor, sbj_names1_uncor = get_data_wrap(fnames1, setup='pet1m3_stag_uncorr')
pe2_uncor, stag2_uncor, sbj_names2_uncor = get_data_wrap(fnames2, setup = 'pet1m3_stag_uncorr')

def stag_all_sbjs(stag):
    s = []; name = []
    for i in range(len(stag)):
        s.append(stag[i]['numeric'].values)
        name.append([str(stag[i].columns[0])] * len(stag[i]))
    return np.hstack(s), np.hstack(name)

def df_setup(stags, setup):
    s_, name_ = stag_all_sbjs(stags[0])
    df1 = pd.DataFrame({'stag':s_, 'name':name_, 'time':np.asarray(['t1']*len(name_))})
    s_, name_ = stag_all_sbjs(stags[1])
    df2 = pd.DataFrame({'stag':s_, 'name':name_, 'time':np.asarray(['t2']*len(name_))})
    #combine stages for t1 with t2
    df = pd.concat([df1, df2])
    df['setup'] = np.asarray(([setup] * len(df)))
    return df

df_corr = df_setup(stags=[stag1, stag2], setup='pet1m3')
df_uncorr = df_setup(stags=[stag1_uncor, stag2_uncor], setup='pet1m3_stag_uncorr')
df = pd.concat([df_corr, df_uncorr])  #big data frame (t1, t2, setup1, setup2)
###########################################################################
'''
#plot 'global' hist stages count from all sbjs included
'''
sns.set(style="darkgrid")
g = sns.FacetGrid(df, row='time', col='setup', margin_titles=True)
axes = g.axes
axes[0,1].set_ylim(0,1000)
axes[0,1].yaxis.set_ticks(np.arange(0,1000, 100))
axes[1,0].xaxis.set_ticks([1,2,3])
bins = len(unique(df['stag']))
g.map(plt.hist, 'stag', color='steelblue', bins=bins, normed=False, ec='black')

###########################################################################
'''
get indiv. count of stages
'''
def get_indiv_stag(this_df):
    nms_ = unique([this_df['name'].iloc[i][:] for i in range(len(this_df))])
    count_store = []
    for n in nms_:
        count_ = this_df[['name', 'stag']][this_df['name']==n].groupby(['stag']).count()
        count_ = count_.rename(columns={'name': 'count'})
        count_ = count_.reindex([1,2,3]).transpose()#Nan is stag not present
        count_['name'] = n
        count_['name_short'] = n[:3]
        count_store.append(count_)
    return pd.concat(count_store, ignore_index=True)

d_corr = get_indiv_stag(df_corr)
d_corr['time'] = [d_corr['name'].iloc[i][4] for i in range(len(d_corr))]
d_uncorr = get_indiv_stag(df_uncorr)
d_uncorr['time'] = [d_uncorr['name'].iloc[i][4] for i in range(len(d_uncorr))]

#plot indic count as heatmap
def plot_heatmap(d_):
    fig, ax = plt.subplots(1,2)
    for i, c in zip([0,1], ['summer','winter']):
        d_sub = d_[[1,2,3]][d_['time'] == str(i+1)]
        im = ax[i].imshow(d_sub, cmap=c)
        ax[i].set_yticks(range(len(d_sub)))
        ax[i].set_yticklabels(d_['name'][d_['time'] == str(i+1)])
        ax[i].set_xticks([0,1,2])
        ax[i].set_xticklabels(['nrem', 'rem', 'w'])
        #cbaxes = fig.add_axes([0.4, 0.9, 0.07, 0.04])
        cb = plt.colorbar(im, ax=ax[i],orientation='horizontal')
    plt.show()
plot_heatmap(d_corr)
