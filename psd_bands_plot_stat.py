'''
process  PSD values: load separately for t1 and t2,
get average psd for given sleep stage, pack into df,
ploting
'''
import os
import numpy as np
import pandas as pd
from config import myload, base_path, paths, chs_incl, subjects, bad_sbjs_1, bad_sbjs_2
from functional import select_class_to_classif, remove_20hz_artif, load_single_append
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.debugger import set_trace
import scipy.stats
import numpy as np, scipy.stats as st
from random import shuffle
import random
from matplotlib.patches import Patch
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

sel_idxs = [1,2,3]
select_single_channel = False #'O2' #'O1' #'O2' #specify channel to select
k = 10  #4 # epochs to average
balance_sbjs_across_groups = False #if True sample 12 estimates from each group (e,g week2 NREM)
stag_names = ['nrem', 'rem', 'wake']

mspe1, mspe_stag1, mspe_names1, _ = load_single_append(path, fnames1, typ='mspet1m3')
mspe2, mspe_stag2, mspe_names2, _ = load_single_append(path, fnames2, typ='mspet1m3')
psd1, psd_stag1, psd_names1, freqs = load_single_append(path, fnames1, typ='psd_v2')
psd2, psd_stag2, psd_names2, freqs = load_single_append(path, fnames2, typ='psd_v2')

assert all([all(mspe_stag1[i] == psd_stag1[i]) for i in range(len(psd_stag1))])
assert all([all(mspe_stag2[i] == psd_stag2[i]) for i in range(len(psd_stag2))])
del (psd_stag1, psd_stag2)

mspe1, psd1, stag1, _ = remove_20hz_artif(mspe1, psd1, mspe_stag1, mspe_names1, freqs, bad_sbjs_1)
mspe2, psd2, stag2, _ = remove_20hz_artif(mspe2, psd2, mspe_stag2, mspe_names2, freqs, bad_sbjs_2)


#drop last  channesl (EOG, ECG) etc
psd1 = [psd1[i][:,:-5,:] for i in range(len(psd1))]
psd2 = [psd2[i][:,:-5,:] for i in range(len(psd2))]
mspe1 = [mspe1[i][:,:-5,:] for i in range(len(mspe1))]
mspe2 = [mspe2[i][:,:-5,:] for i in range(len(mspe2))]

mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)
psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

def av_channels_epochs(data, stag, what_stag, names):
    '''
    Agragate across epochs and/or channels
    '''
    #select data corr with given sleep stage s
    np.random.seed(123)
    av_stag = [data[i][:,:, np.where(stag[i]['numeric'] == what_stag)] for i in range(len(data))]
    names_ss = [ names[i] for i in range(len(av_stag)) if av_stag[i].size != 0 ]
    if k != 'all': #average k number of epochs
        ep_i = [ av_stag[i].shape[3] for i in range(len(av_stag)) ] #count how many epochs we have for given ss
        #randmoly sample epochs
        ep_sample_i = [np.random.choice(ep_i[i], size=k, replace=False) if ep_i[i] > k  else [] if ep_i[i] == 0 else np.arange(ep_i[i]) for i in range(len(ep_i)) ]
        #average epochs using random index
        av_stag =[ np.where(len(ep_sample_i[i]) > 0 , np.median(av_stag[i][:,:,:,ep_sample_i[i]], 3), np.nan) for i in range(len(av_stag)) ]
    else: #average all epochs
        av_stag =[ np.where(av_stag[i].size != 0, np.nanmedian(av_stag[i], 3), np.full(av_stag[i].shape[:3], np.nan)) for i in range(len(av_stag))]

    #average or select channels
    if select_single_channel is not False:
        ch_idx = np.where(np.asarray(chs_incl)==select_single_channel)[0]
        av_stag = [np.squeeze(np.asarray(av_stag[i][:,ch_idx,:])) for i in range(len(av_stag))]
    else:
        av_stag = [np.squeeze(np.nanmedian(av_stag[i], 1)) for i in range(len(av_stag))]

    return av_stag, names_ss

def iterate_stages_getdict(data, stag, sel_idxs, names):
    mydict = {}
    freq_bins_len = data[0].shape[0]
    for s in sel_idxs:
        channels_aved, found_names = av_channels_epochs(data, stag, s, names)
        channels_aved_red = [ channels_aved[i] for i in range(len(channels_aved)) if not np.isnan(channels_aved[i][0]) ]
        #sample 12, count(week2 NREM) = 12
        if balance_sbjs_across_groups:
            random.Random(12).shuffle(channels_aved_red)
            channels_aved_red = channels_aved_red[:12]
            found_names =  [[] for i in range(12)]
        #when balancing skip names
        mydict[str(s)] =(channels_aved_red, found_names)
    return mydict



mydict1 = iterate_stages_getdict(psd1, stag1, sel_idxs, psd_names1)
mydict2 = iterate_stages_getdict(psd2, stag2, sel_idxs, psd_names2)

def wrap_together_df(mydict):
    nrem = np.vstack(mydict['1'][0])
    rem = np.vstack(mydict['2'][0])
    wake = np.vstack(mydict['3'][0])
    def built_df(stag, data, names):
        df = pd.DataFrame(data, columns = range(data.shape[1]))
        df['stag'] = [stag] * data.shape[0]
        df['sbj_id'] = names
        return df
    dfs = []
    nrem_ns = mydict['1'][1]
    rem_ns = mydict['2'][1]
    wake_ns = mydict['3'][1]
    for s, d, n  in zip(['nrem', 'rem', 'wake'], [nrem, rem, wake], [nrem_ns, rem_ns, wake_ns]):
        df_ = built_df(s, d, n)
        dfs.append(df_)

        df = pd.concat(dfs, 0)
    return df
df1 = wrap_together_df(mydict1)
df1 = df1.dropna()

##################################
band = 'slow'
freq_min, freq_max = (4,12) # numpy like subseting( incl, excl)

save_name = 'psd_theta_alpha_4_11Hz.csv'
#################################

df1[band] = df1.iloc[:, freq_min:freq_max].sum(1)
df1_long = pd.melt(df1.loc[:, ['stag', 'sbj_id', band]], id_vars=['stag', 'sbj_id'])
df1_long['time'] = '2week'

df2 = wrap_together_df(mydict2)
df2 = df2.dropna()
df2[band] = df2.iloc[:, freq_min:freq_max].sum(1)
df2_long = pd.melt(df2.loc[:, ['stag', 'sbj_id', band]], id_vars=['stag', 'sbj_id'])
df2_long['time'] = '5week'


df_long = pd.concat([df1_long, df2_long])
#df_long.to_csv(save_name) #save



##################################################################
# PLot
df_long = df_long.reset_index()
df = df_long
df = df.dropna()
df['value'] = df['value'] * 100
fig, ax = plt.subplots()
box = sns.boxplot(x='stag', y='value', hue='time',data=df, linewidth=3, \
                ax=ax, whis=[5, 95], showfliers=False, dodge=True)

def modif_box(axes, facecolor, edgecolor, idxs):
    for i in idxs:
        mybox = box.artists[i]
        mybox.set_facecolor(facecolor)
        mybox.set_edgecolor(edgecolor)
        # Each box has several associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*5,i*5+5):
            ax = plt.gca()
            line = ax.lines[j]
            line.set_color(edgecolor)
            line.set_mfc(edgecolor)
            line.set_mec(edgecolor)
    return

modif_box(box, 'white', 'black', [0,2,4])
modif_box(box, 'white', 'red', [1,3,5])

sns.swarmplot(x='stag', y='value', hue='time',data=df, split=True, color='black', \
            size=4, alpha=0.7, ax=ax)

legend_elements = [Patch(facecolor='white', edgecolor='black',
                         label='week 2', linewidth=3),
                    Patch(facecolor='white', edgecolor='red',
                        label='week 5', linewidth=3)]
ax.legend(handles=legend_elements, loc='lower right')
#ax.set(ylabel= 'MSPE(scale=5)', xlabel='', xticklabels= ['NREM', 'REM', 'WAKE'], ylim=[1.2, 1.68])
ax.set(xlabel='', xticklabels= ['NREM', 'REM', 'WAKE'])#, ylim=[75, 100])  #ylim=[-0.6, 5]) # ylim=[75, 100]
plt.show()








'''
def get_band(mydict, stage, band):
    d = mydict[stage][0]
    print(len(d))
    band_power = [ d[i][band].sum() for i in range(len(d)) ]
    return(band_power)

bands = {'slow':(1,8), 'fast':(10,18)}
# Init empty dict
d1  = {'slow': {'1': [], '2': [], '3': []},
            'fast': {'1': [], '2': [], '3': []} }

d2  = {'slow': {'1': [], '2': [], '3': []},
            'fast': {'1': [], '2': [], '3': []} }

# Sum values for a given bands
for b_name, b in bands.items():
    for s in ['1', '2', '3']:
        temp1 = get_band(mydict1, stage=s, band=slice(*b))
        temp1.extend([None]*36) # add None of sbj missing, count REM week2 = 36 (max)
        temp1 = temp1[:36]
        d1[b_name][s] = temp1
        temp2 = get_band(mydict2, stage=s, band=slice(*b))
        temp2.extend([None]*36)
        temp2 = temp2[:36]
        d2[b_name][s] = temp2

df1 = pd.DataFrame.from_dict({(i,j): d1[i][j]
                           for i in d1.keys()
                           for j in d1[i].keys()},
                       orient='columns')
df2 = pd.DataFrame.from_dict({(i,j): d2[i][j]
                           for i in d2.keys()
                           for j in d2[i].keys()},
                       orient='columns')
'''
