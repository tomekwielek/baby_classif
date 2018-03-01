import os
import mne
from config import raw_path, myload, base_path
from mne import io
from mne import EpochsArray, EvokedArray
from matplotlib import pyplot as plt
from functional import (load_single_append, select_stages_to_classif, merge_stages,
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
    pe, stag, sbj_names = load_single_append(base_path, fnames, typ = setup)
    pe, stag = select_stages_to_classif(pe, stag, sel_idxs=sel_idxs)
    if merge:
        stag_m = [stag[i].iloc[:,[0]].replace(mapper, inplace=False).astype(float) \
            for i in range(len(stag))]
    elif merge == False:
        stag_m = stag
    return pe, stag_m, sbj_names

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
        indiv_['av', str(idx)] = [np.mean(array_[i]) for i in range(len(stag))]
        indiv_['std', str(idx)] = [np.std(array_[i]) for i in range(len(stag))]
        indiv_['name_id', str(idx)]  = [stag[i].columns[0] for i in range(len(stag))]
    return indiv_
#load data, merge stages (e.g 1 and 2 as single sleep) for merge ==True
pe1, stag1, sbj_names1 = get_data_wrap(fnames1, merge=merge)
pe2, stag2, sbj_names2 = get_data_wrap(fnames2, merge=merge)
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
    # what stages we have
    #TODO when merge then col names change; condition needed
    if merge:
        uniq_stag = map(lambda x: x.iloc[:,0].unique().tolist(), stag)
    else:
        uniq_stag = map(lambda x: x.iloc[:,1].unique().tolist(), stag)
    uniq_stag = unique(reduce(lambda x,y: x+y, uniq_stag))
    uniq_stag = [str(int(s)) for s in uniq_stag]
    av_melt = pd.melt(av, value_vars = uniq_stag, id_vars='time_id', var_name='stag')
    av_melt.dropna(inplace = True)
    return av, av_melt

av, av_melt, = prepare_data(peall, stagall)
av_melt['value'] = pd.to_numeric(av_melt['value'])# get numerc type
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
#plot_descriptive(av, uniq_stag, count)

sns.factorplot(x='stag', hue='time_id', y='value', data=av_melt,ci=95, kind='point')
sns.factorplot(x='stag', hue='time_id', y='value', data=av_melt,ci=95, kind='strip')

av_melt.groupby(['time_id', 'stag'])
'''
import scipy.stats as stats
#paired t test
t1 = np.asarray(av_melt['value'][ av_melt['time_id']=='1' |
                                 av_melt['stag'] =='1' ])
t2 = np.asarray(av_melt['value'][av_melt['time_id'] == '2'])
stats.ttest_ind(t1,t2)
#stats.ttest_1samp(t1-t2, popmean=0) #another option
stats.ttest_ind(t1, t2)
'''


# ANOVA & posthoc
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as multi
formula = 'value ~ C(stag) + C(time_id) + C(stag):C(time_id)'
mod =  ols(formula, av_melt).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)
print aov_table

#pairwise posthoc
mc1=multi.MultiComparison(av_melt['value'],av_melt['stag'])
res1=mc1.tukeyhsd()
print res1.summary()




'''
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)

anova = AnovaRM(av, depvar='value', subject='id', within=['stag'])
fit = anova.fit()
fit.summary()

mc = MultiComparison(av['value'], av['stag'])
results = mc.tukeyhsd()
results.plot_simultaneous()
print results
print mc.groupsunique
'''
