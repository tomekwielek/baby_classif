'''
process  MSPE values: load separately for t1 and t2,
get average \ for given sleep stage, pack into df,
ploting
'''
import os
import numpy as np
import pandas as pd
from config import myload, base_path, paths
from functional import (select_class_to_classif,  load_single_append)
from config import myload, subjects
from functional import select_class_to_classif
import matplotlib.pyplot as plt

sel_idxs = [1,2,3]
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
setups= ['mspet1m3']
store_scores = dict()
store_pval = dict()

for setup in setups:
    pe1, stag1, _, _ = load_single_append(path, fnames1, typ = setup)
    pe2, stag2, _, _ = load_single_append(path, fnames2, typ = setup)
    pe1, stag1 = select_class_to_classif(pe1, stag1, sel_idxs=sel_idxs)
    pe2, stag2 = select_class_to_classif(pe2, stag2, sel_idxs=sel_idxs)
    pe1 = [pe1[i][0:5,:,:] for i in range(len(pe1))] #drop higher scales
    pe2 = [pe2[i][0:5,:,:] for i in range(len(pe2))]
    #drop last two channesl (EOG and ECG)
    #rel_pe1 = [rel_pe1[i][:,:-2,:] for i in range(len(rel_pe1))]
    #rel_pe2 = [rel_pe2[i][:,:-2,:] for i in range(len(rel_pe2))]


def av_channels(data, stag, what_stag):
    '''
    avearge across channels only
    '''
    if data[0].ndim == 2: #pe
        av_stag = [data[i][:, stag[i]['numeric'] == what_stag].mean(0) for i in range(len(data))]
    elif data[0].ndim == 3: #mspe
        av_stag = [data[i][..., stag[i]['numeric'] == what_stag] for i in range(len(data))]
        av_stag = [av_stag[i].mean(1) for i in range(len(av_stag))]
    return av_stag

def av_epochs(data, stag, sel_idxs):
    mydict = {}
    for s in sel_idxs:
        channels_aved = av_channels(data, stag, s)
        #avearge epochs of a given stage
        ep_mean = [ np.nanmean(channels_aved[i], 1) for i in range(len(channels_aved)) ]
        ep_std = [ np.nanstd(channels_aved[i], 1) for i in range(len(channels_aved)) ]
        mydict[str(s)] =(ep_mean, ep_std)
    return mydict

mydict1 = av_epochs(pe1, stag1, sel_idxs)
mydict2 = av_epochs(pe2, stag2, sel_idxs)


def wrap_together_df(mydict):
    nrem = np.vstack(mydict['1'][0]) # 0 index is mean, 1 index is std
    rem = np.vstack(mydict['2'][0])
    wake = np.vstack(mydict['3'][0])

    def built_df(stag, data):
        df = pd.DataFrame(data, columns = range(data.shape[1]))
        df['stag'] = [stag] * data.shape[0]
        return df
    dfs = []
    for s, d  in zip(['nrem', 'rem', 'wake'], [nrem, rem, wake]):
        df_ = built_df(s, d)
        dfs.append(df_)

    df = pd.concat(dfs, 0)
    return df

df1 = wrap_together_df(mydict1)
df1 = df1.dropna()
df1_long = pd.melt(df1, id_vars='stag')
df1_long['time'] = ['2week'] * len(df1_long)
df2 = wrap_together_df(mydict2)
df2 = df2.dropna()
df2_long = pd.melt(df2, id_vars='stag')
df2_long['time'] = ['5week'] * len(df2_long)


################## seperate plot for t1 and t2, 'stag' hues the plots ########################
'''
import seaborn as sns
sns.factorplot(x='variable', y = 'value', hue='stag', data=df1_long)
#plt.ylim(-130, -90)
old_ticks = plt.gca().get_xticks()
plt.xticks(old_ticks[::3], np.around(freqs,1)[::3])

sns.factorplot(x='variable', y = 'value', hue='stag', data=df2_long)
#plt.ylim(-130, -90)
old_ticks = plt.gca().get_xticks()
plt.xticks(old_ticks[::3], np.around(freqs,1)[::3])
'''


################# cat t1 with t2, 'time' hues the plots ######################################
df_long = pd.concat([df1_long, df2_long])

def plot_stages_time_hued(df):
    import seaborn as sns
    fig, axes = plt.subplots(1,3, sharey=True, figsize = (10,5))
    for axi, s in enumerate(['nrem', 'rem', 'wake']):
        sns.pointplot(x='variable', y = 'value', hue='time', ci=95, \
                        data=df[df['stag']==s], ax = axes[axi])
        axes[axi].set_title(s.upper())
        axes[axi].set_xlabel('Scale')
        axes[axi].set_xticklabels([1,2,3,4,5])
        axes[axi].set_ylabel('Permutation Entropy')
    plt.show()


plot_stages_time_hued(df_long)





'''
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize = (10,5))
sns.regplot(x = 'variable', y = 'value', ci=95, \
                data=df_long[df_long['time']=='2week'], ax=ax1 )

sns.regplot(x = 'variable', y = 'value', ci=95, \
                data=df_long[df_long['time']=='5week'], ax=ax2 )
'''
