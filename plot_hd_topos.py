import os
import numpy as np
from config import myload, base_path, paths, bad_sbjs_1, bad_sbjs_2
from mne import io
import pickle
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import mne
from config import stag_fname, length_issue_sbjs, match_names27, match_names35, markers35
import pandas as pd
from functional import (my_rename_col, map_stag_with_raw, load_single_append,
                            select_class_to_classif, remove_20hz_artif)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.rcParams.update({'font.size': 14})


mff_single = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\104_2_20150113.mff' #for raw info
base_path = 'H:\\BABY\\working\\subjects'
match_names = match_names35 + match_names27
match_names = [ match_names[i].split('_S')[0] for i in range(len(match_names)) ]
match_names_1 = [m for m in match_names if '_1' in m]
match_names_2 = [m for m in match_names if '_2' in m]
sel_idxs = [1,2,3]
k= 'all' #4 #epochs to average
stages_mapp = {1:'NREM', 2:'REM', 3:'WAKE'}


mspe1, mspe_stag1, mspe_names1, _ = load_single_append(base_path, match_names_1, typ='mspet1m3')
mspe2, mspe_stag2, mspe_names2, _ = load_single_append(base_path, match_names_2, typ='mspet1m3')
psd1, psd_stag1, psd_names1, freqs = load_single_append(base_path, match_names_1, typ='psd_hd')
psd2, psd_stag2, psd_names2, freqs = load_single_append(base_path, match_names_2, typ='psd_hd')

assert all([all(mspe_stag1[i] == psd_stag1[i]) for i in range(len(psd_stag1))])
assert all([all(mspe_stag2[i] == psd_stag2[i]) for i in range(len(psd_stag2))])
del (psd_stag1, psd_stag2)

mspe1, psd1, stag1, _ = remove_20hz_artif(mspe1, psd1, mspe_stag1, mspe_names1, freqs, bad_sbjs_1)
mspe2, psd2, stag2, _ = remove_20hz_artif(mspe2, psd2, mspe_stag2, mspe_names2, freqs, bad_sbjs_2)


mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)
psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

# raw to get chs info
m = mne.channels.read_montage('GSN-HydroCel-129')
raw = mne.io.read_raw_egi(mff_single, preload=True, include=markers35).pick_types(eeg=True)
#raw.ch_names.pop() #del last
info = mne.create_info(ch_names = raw.ch_names, sfreq=125., ch_types='eeg', montage=m)

def av_epochs(data, stag, what_stag):
    '''
    Function agragates across epochs
    '''
    np.random.seed(123)
    av_stag = [data[i][:,:, np.where(stag[i]['numeric'] == what_stag)] for i in range(len(data))]
    if k != 'all': #average k number of epochs
        ep_i = [ av_stag[i].shape[3] for i in range(len(av_stag)) ] #count how many epochs we have for given ss
        #randmoly sample epochs
        ep_sample_i = [np.random.choice(ep_i[i], size=k, replace=False) if ep_i[i] > k  else [] if ep_i[i] == 0 else np.arange(ep_i[i]) for i in range(len(ep_i)) ]
        #average epochs using random index
        av_stag =[ np.where(len(ep_sample_i[i]) > 0 , np.median(av_stag[i][:,:,:,ep_sample_i[i]], 3), np.nan) for i in range(len(av_stag)) ]
    else: #average all epochs
        av_stag =[ np.where(av_stag[i].size != 0, np.nanmedian(av_stag[i], 3), np.full(av_stag[i].shape[:3], np.nan)) for i in range(len(av_stag))]
    return av_stag

def plot_topo_psd(psd_to_plot, stag, what_stag, info, fmin, fmax, week=None):
    fig, ax = plt.subplots(figsize=(3,4))
    psd_ = av_epochs(psd_to_plot, stag, what_stag=what_stag)
    psd_data = [ psd_[i] for i in range(len(psd_)) if not np.isnan(psd_[i][0]).any() ]
    #set_trace()
    psd_data = np.asarray(psd_data).mean(0) #mean sbj
    freqs_mask = np.logical_and(freqs>=fmin, freqs<=fmax) #mask to select bins

    psd_data = psd_data[freqs_mask, :, :].sum(0)[np.newaxis, :,:] #sum power for selected bins

    psd_data = psd_data.transpose([1,0, 2])
    psd_data = psd_data*100. #convert to %
    tfr = mne.time_frequency.AverageTFR(info, psd_data, times=[1], freqs=[1], nave=1)

    tfr.plot_topomap(axes=ax, cbar_fmt='%3.1f', vmin=80, vmax=100,  unit='Relative\npower\n[%]',
                    contours=0)
    ax.set_title(week)
    plt.tight_layout()
    plt.show()

what_stag=3

# freq ranges (0,3) and (12,29)
plot_topo_psd(psd1, stag1, what_stag=what_stag, info=info, fmin=1, fmax=3, week= 'week2')
plot_topo_psd(psd2, stag2, what_stag=what_stag, info=info, fmin=1, fmax=3, week= 'week5')

def get_av_psd(psd_to_plot, stag, what_stag, info, fmin, fmax, week=None):
    psd_ = av_epochs(psd_to_plot, stag, what_stag=what_stag)
    psd_data = [ psd_[i] for i in range(len(psd_)) if not np.isnan(psd_[i][0]).any() ]
    #set_trace()
    psd_data = np.mean(psd_data, 0) #mean/med sbj
    freqs_mask = np.logical_and(freqs>=fmin, freqs<=fmax) #mask to select bins
    psd_data = psd_data[freqs_mask, :, :].sum(0)[np.newaxis, :,:] #sum power for selected bins
    psd_data = psd_data.transpose([1,0, 2])
    psd_data = psd_data*100. #convert to %
    return psd_data


# Contrast for avergae PSD
av2 = get_av_psd(psd1, stag1, what_stag=what_stag, info=info, fmin=4, fmax=11, week='week2')
av5 = get_av_psd(psd2, stag2, what_stag=what_stag, info=info, fmin=4, fmax=11, week='week5' )
contrast = av2- av5
fig, ax = plt.subplots(figsize=(3,4))
tfr = mne.time_frequency.AverageTFR(info, contrast, times=[1], freqs=[1], nave=1)
tfr.plot_topomap(axes=ax, cbar_fmt='%3.1f', unit='Relative\npower\n[%]', vmin=-2.5, vmax=2.5, # vmin=-4.5, vmax=4.5,
                contours=0) #vmin=-2.5, vmax=2.5,
#ax.set_title(stages_mapp[what_stag] + ' ' + 'week2 - week5')
ax.set_title('week2 - week5')

plt.tight_layout()
plt.show()
