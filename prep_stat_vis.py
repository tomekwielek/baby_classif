import os
import mne
from config import raw_path, myload, base_path
from mne import io
from mne import EpochsArray, EvokedArray
from matplotlib import pyplot as plt
from functional import (load_single_append, select_class_to_classif, merge_stages,
                         count_stag)
import pandas as pd
import numpy as np
import seaborn as sns

setup = 'pet1m3'
fnames =  os.listdir(base_path)

fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')]
mapper = {'N' : 1, 'R' : 1, 'W' : 3} #for stages merging
sel_idxs = [1, 2, 3]
merge = False

def get_data_wrap(fnames, merge):
    freqs = None
    if setup != 'psd': # if setup refers to some pe-s
        pe, stag, sbj_names, _ = load_single_append(base_path, fnames, typ = setup)
    elif setup == 'psd':
        pe, stag, sbj_names, freqs = load_single_append(base_path, fnames, typ = setup)
    pe, stag, _ = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)
    if merge:
        stag_m = [stag[i].iloc[:,[0]].replace(mapper, inplace=False).astype(float) \
            for i in range(len(stag))]
    elif merge == False:
        stag_m = stag
    return pe, stag_m, sbj_names, freqs

def plot_mytopomap(pe, indiv_stag):
    raw = io.read_raw_edf(raw_path + '104_2_correctFilter_2heogs_ref100.edf',
                preload=True)
    raw.set_channel_types({'ECG':'ecg', 'EMG':'emg', 'VEOG':'eog', 'HEOG_l':'eog'})
    m = mne.channels.read_montage('standard_1020')
    raw = raw.set_montage(m)
    raw.pick_types(eeg=True, ecg=True, emg=True, eog=True)
    info = raw.info
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pe[:6,], info, vmin=pe.min(), vmax=pe.max(), show=False,
                        axes=ax)
    fig.tight_layout()
    fig.suptitle('Sleep stage %s' %indiv_stag)
    fig.show()
'''
get pe values for individual ss, for each subject
'''
def pe_for_indiv_stag(pe, stag, sel_idxs):
    indiv_= dict()
    for idx in sel_idxs:
        #get list of pe-s for given stag
        array_ = [pe[i][:, np.where(stag[i] == idx)[0]] for i in range(len(stag))]
        indiv_['array_', str(idx), ] = array_
        indiv_['av', str(idx)] = [np.mean(array_[i]) for i in range(len(stag))] #av epochs AND chns
        indiv_['std', str(idx)] = [np.std(array_[i]) for i in range(len(stag))]
        indiv_['name_id', str(idx)]  = [stag[i].columns[0] for i in range(len(stag))]
        indiv_['avchs', str(idx)] = [np.mean(array_[i],0) for i in range(len(stag))]
    return indiv_
#load data, merge stages (e.g 1 and 2 as single sleep) for merge ==True
pe1, stag1, sbj_names1, _ = get_data_wrap(fnames1, merge=merge)
pe2, stag2, sbj_names2, _ = get_data_wrap(fnames2, merge=merge)
if merge: #update sel_idxs if stages were merged
    sel_idxs = list(unique(mapper.values()))

peall= pe1 + pe2 #merge data from t1, t2
stagall = stag1 + stag2
'''
prepara data for statistics; built df, drop nan
'''
#note fnames1 (first recording) only a few subjects show all 3 sleep stages
def prepare_data(pe, stag):
    d_ind = pe_for_indiv_stag(pe, stag, sel_idxs) #get nested dict
    mi_df = pd.DataFrame(d_ind).transpose() #get MultiIndex-ed df
    av = pd.DataFrame(mi_df.loc['av'].transpose())

    av['name_id'] = mi_df.loc['name_id'].iloc[0,:]
    av['time_id'] = [av['name_id'].iloc[i][4] for i in range(len(av))]
    #TODO when merge then col names change; condition needed
    if merge:
        uniq_stag = map(lambda x: x.iloc[:,0].unique().tolist(), stag)
    else:
        uniq_stag = map(lambda x: x.iloc[:,1].unique().tolist(), stag)
    # what stages we have
    uniq_stag = np.unique(reduce(lambda x,y: x+y, uniq_stag))
    uniq_stag = [str(int(s)) for s in uniq_stag]
    av_melt_with_time_id = pd.melt(av, value_vars = uniq_stag, id_vars='time_id', var_name='stag')
    av_melt_with_name_id = pd.melt(av, value_vars = uniq_stag, id_vars='name_id', var_name='stag')
    av_melt = pd.concat([av_melt_with_time_id, av_melt_with_name_id], axis=1, join='inner')
    av_melt = av_melt.loc[:,~av_melt.columns.duplicated()] #drop duplicated columns
    av_melt['name_id_short'] =  [av_melt['name_id'].iloc[i][0:3] for i in range(len(av_melt))]
    #av_melt.dropna(inplace = True)
    return av, av_melt

'''plot distibutin, normal check'''
def plot_descriptive(av, uniq_stag, count):
    import seaborn as sns
    import statsmodels.api as sm
    fig, axes = plt.subplots(3,1, figsize = (6,14))
    box = av.boxplot(['value'], by=['stag', 'time_id'], ax=axes[0])

    axes[0].set_ylim([1.35,1.55])
    axes[0].set_title('N sbj = %s' %str(count))
    for s in uniq_stag:
        d = av['value'][av['stag']==s]
        sns.distplot(d, hist=False, rug=True, ax =axes[1])
        sm.qqplot(d, ax = axes[2])
    plt.suptitle('')
    plt.show()

av, av_melt, = prepare_data(peall, stagall)
av_melt['value'] = pd.to_numeric(av_melt['value'])# get numerc type

#av.to_csv('df_w.csv')

#sns.factorplot(x='stag', hue='time_id', y='value', data=av_melt,ci=95, kind='point')
#sns.factorplot(x='stag', hue='time_id', y='value', data=av_melt,ci=95, kind='strip')
#plt.show()

'''
# NO averaging over epochs
mydict = pe_for_indiv_stag(peall, stagall, sel_idxs) #nested dict storing array, av ,std
avchs = {key : value for key, value in mydict.iteritems() if 'avchs' in key} #select only avs over chs
names =  {key : value for key, value in mydict.iteritems() if 'name_id' in key} #select only name_id
t_ = names.copy()
avchs.update(t_) #merge dicts
del t_

midf = pd.DataFrame(avchs).transpose() #get MultiIndex-ed df
df = pd.DataFrame(midf.loc['avchs'].transpose())
#df = pd.DataFrame(midf.loc['avchs'])
df['name_id'] = midf.loc['name_id'].iloc[0,:]
df['time_id'] = [df['name_id'].iloc[i][4] for i in range(len(df))]

uniq_stag = map(lambda x: x.iloc[:,1].unique().tolist(), stagall)
uniq_stag = np.unique(reduce(lambda x,y: x+y, uniq_stag))
uniq_stag = [str(int(s)) for s in uniq_stag]
df_melt_with_time_id = pd.melt(df, value_vars = uniq_stag, id_vars='time_id', var_name='stag')
df_melt_with_name_id = pd.melt(df, value_vars = uniq_stag, id_vars='name_id', var_name='stag')
df_melt = pd.concat([df_melt_with_time_id, df_melt_with_name_id], axis=1, join='inner')
df_melt = df_melt.loc[:,~df_melt.columns.duplicated()] #drop duplicated columns
df_melt['name_id_short'] =  [df_melt['name_id'].iloc[i][0:3] for i in range(len(df_melt))]
'''
#create long format for pe avergaed  across channels AND k number of randomly sampled epochs
def pe_stag_time_names(pe, stag, sel_idxs):
    k = 5 #sample size of sample epochs
    pes, ss, time_id, name_id, name_id_short = [[] for i in range(5)]
    for idx in sel_idxs:
        pes_ = pe[:, np.where(stag == idx)[0]].mean(0) #av channels
        nb_repl = len(pes_)
        if nb_repl >= k:
            pes_ = np.random.choice(pes_, size=k, replace=False)
        elif nb_repl == 0:
            pes_ = np.nan
        pes.append(np.nanmean(pes_))
        ss.append(idx)
        time_id.append(stag.columns[0][4])
        name_id.append(stag.columns[0])
        name_id_short.append(stag.columns[0][0:3])

    mydict = {'value':pes,
            'stag':np.hstack(ss), 'time_id':np.hstack(time_id), \
            'name_id':np.hstack(name_id), 'name_id_short':np.hstack(name_id_short)}
    df = pd.DataFrame(mydict)
    return df

list_dfs = [pe_stag_time_names(peall[i], stagall[i], sel_idxs) for i in range(len(peall))]

df = pd.concat(list_dfs)
sns.factorplot(x='stag', y='value', hue='time_id', data=df)
#df.to_csv('df_l_eps.csv')
