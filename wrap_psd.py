'''
process  PSD values: load separately for t1 and t2,
get average psd for given sleep stage, pack into df,
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
import seaborn as sns

sel_idxs = [1,2,3]
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
setups= ['psd']
store_scores = dict()
store_pval = dict()

for setup in setups:
    psd1, stag1, _, freqs = load_single_append(path, fnames1, typ = setup)
    psd2, stag2, _, freqs = load_single_append(path, fnames2, typ = setup)
    psd1, stag1 = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
    psd2, stag2 = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

    # Relative power see: Xiao(2018)
    rel_psd1 = [psd1[i] / np.abs(np.sum(psd1[i], 0)) for i in range(len(psd1))]
    rel_psd1 = [ np.log10(rel_psd1[i]) for i in range(len(psd1)) ]
    rel_psd2 = [psd2[i] / np.abs(np.sum(psd2[i], 0)) for i in range(len(psd2))]
    rel_psd2 = [ np.log10(rel_psd2[i]) for i in range(len(psd2)) ]

    #drop last two channesl (EOG and ECG)
    #rel_psd1 = [rel_psd1[i][:,:-2,:] for i in range(len(rel_psd1))]
    #rel_psd2 = [rel_psd2[i][:,:-2,:] for i in range(len(rel_psd2))]

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
    k = 10 # no of epochs to sample
    freq_bins_len = data[0].shape[0]
    for s in sel_idxs:
        channels_aved = av_channels(data, stag, s)
        # count n epochs
        ep_i = [ channels_aved[i].shape[1] for i in range(len(channels_aved)) ]
        # random index of k epochs if no epochs then Nan
        ep_sample_i = [np.random.choice(ep_i[i], size=k, replace=False) if ep_i[i] > k  else np.NaN if ep_i[i] == 0 else np.arange(ep_i[i]) for i in range(len(ep_i)) ]
        # sample using random index if no epochs then array of Nan-s
        ep_mean = [ np.nanmean(channels_aved[i][:, ep_sample_i[i]], 1) if not isinstance(ep_sample_i[i],float) else np.full([freq_bins_len,], np.nan) for i in range(len(ep_i)) ]
        #ep_mean = ep_mean[~numpy.isnan(ep_mean)]
        mydict[str(s)] =(ep_mean)
    return mydict

# keys repr. ss and values av. psd e.g:
# mydict['1'][0] repr. list (all sbjs) with av. psd for ss. '1'
mydict1 = av_epochs(rel_psd1, stag1, sel_idxs)
mydict2 = av_epochs(rel_psd2, stag2, sel_idxs)

# wrap_together_df stacks averaged epochs across subjects, returned df.shape is
#  #N*len('1', '2', '2') x #freq_bins
def wrap_together_df(mydict):
    nrem = np.vstack(mydict['1'])
    rem = np.vstack(mydict['2'])
    wake = np.vstack(mydict['3'])

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
df1['time'] = ['2week'] * len(df1)
df1_long['time'] = ['2week'] * len(df1_long)
df2 = wrap_together_df(mydict2)
df2 = df2.dropna()
df2_long = pd.melt(df2, id_vars='stag')
df2['time'] = ['5week'] * len(df2)
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
df = pd.concat([df1, df2])

def plot_stages_time_hued(df):
    import seaborn as sns
    fig, axes = plt.subplots(1,3, sharey=True, figsize = (10,5))
    for axi, s in enumerate(['nrem', 'rem', 'wake']):
        sns.pointplot(x='variable', y = 'value', hue='time', ci=95, join=True, scale=0.5,units ='stag',\
                        data=df[df['stag']==s], ax = axes[axi], n_boot=1000,capsize =1.4, errwidth =.8, \
                        palette = sns.color_palette('Dark2'))
        old_ticks = axes[axi].get_xticks()
        axes[axi].set(xticks=old_ticks[::10], xticklabels=freqs.astype(int)[::10])
        axes[axi].set_title(s.upper(), fontsize=20)
        axes[axi].set_xlabel('Freqs [Hz]', fontsize=16)
        axes[axi].yaxis.set_ticklabels([])
        axes[0].set_ylabel('Log Relative Power', fontsize=20)
        axes[axi].legend_.remove()
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    plt.legend(fontsize=20)
    plt.show()

plot_stages_time_hued(df_long)

def log_freqs(df):
    f = df['variable'].astype('int').as_matrix()
    f = np.ma.log(f)
    f= np.ma.filled(f,0)
    df['variable'] = f
    return df


#df_long_log = log_freqs(df_long)
#plot_stages_time_hued(df_long_log)

#no spliting based on ss
#sns.pointplot(x='variable', y = 'value', hue='time', ci=95, \
#                data=df_long)


'''
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize = (10,5))
sns.regplot(x = 'variable', y = 'value', ci=95, \
                data=df_long[df_long['time']=='2week'], ax=ax1 )

sns.regplot(x = 'variable', y = 'value', ci=95, \
                data=df_long[df_long['time']=='5week'], ax=ax2 )
'''

#df.to_csv('df_psd.csv')
#df_long.to_csv('df_long_psd.csv')
