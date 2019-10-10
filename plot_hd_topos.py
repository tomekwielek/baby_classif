'''
(1) Read HD Epochs
(2) Compute welch PSD
(3) Aggregate epochs using random re-sampling
(4) PLot PSD with cluster perm. stat
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload, mysave
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
from functional import write_pickle, read_pickle
from scipy.stats import sem
from mne.stats import spatio_temporal_cluster_test, permutation_cluster_test

matplotlib.rcParams.update({'font.size': 12,'xtick.labelsize':8, 'ytick.labelsize':8})
np.random.seed(12)

path = 'H:\\BABY\\working\\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

events_id = ['N', 'R', 'W']
events_id_map = {'N':1, 'R':2, 'W':3}
store = {'week2':[], 'week5':[]}
store_psd =  {'week2':[], 'week5':[]}
store_event =  {'week2':[], 'week5':[]}
store_name = {'week2':[], 'week5':[]}

m = mne.channels.read_montage(kind='GSN-HydroCel-128')
epoch = myload(typ='epoch_hd', sbj='108_2')
info = epoch.info
del epoch

colors = ['black', 'red']
fmin,fmax = (10, 14) # SET FREQS

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        #empty folder means missing data, see exclusion criteria
        if len(os.listdir(os.path.dirname(paths(typ='epoch_hd', sbj=sbj)))) == 0:
            continue
        else:
            epoch = myload(typ='epoch_hd', sbj=sbj)
            epoch = epoch[events_id]
            epoch.set_montage(m)
            psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128,
                                                    picks='all', n_overlap=64) 
            
            psds = 10. * np.log10(psds) #dB
            freqs = freqs.astype(int)
            n_freqs = len(freqs)
            #store[time].append(epoch_clean)
            store_psd[time].append(psds)
            store_event[time].append(epoch.events[:,-1])
            store_name[time].append(sbj)

#write_pickle((store_psd, store_event, store_name, freqs), 'hd_psds_stages_names_freqs.txt')

store_psd, store_event, store_name, freqs = read_pickle('hd_psds_stages_names_freqs.txt')
n_freqs = len(freqs)

# Plot power spectrum density (bootrstaped averaged epochs )
def my_bootstraper(data, ch_indices, repetions=1000, n=10):
    '''
    Bootstraped averaging of n epochs (epochs per sleep stage)
    data[i].shape = n_epochs x n_chs x freqs where i=sbj index
    '''
    #if fmin is not None:
    #    freqs_mask = np.logical_and(freqs>=fmin, freqs<=fmax) #mask to select bins
    #else:
    #    freqs_mask = [True] * len(freqs)                   
    np.random.seed(None) # randomly initialize the RNG
    store_repet = np.zeros([repetions, len(data), 129, len(freqs)]) #129 channels
    for i in range(repetions): 
        #store = []
        for j, d_ in enumerate(data): # data is a list of psds (over sbjs)
            count = d_.shape[0]    
            if count == 0:
                store_repet[i,j,:] = [np.nan] * len(freqs)
            else:
                sample = np.min([n, count]) # if n<count mean over count
                epoch_indices = np.random.choice(count, sample)
                d_ = d_[epoch_indices, :, :] #sample epochs
                store_repet[i,j,:] = d_.mean((0)) #mean epochs 
    return store_repet

# PLOT psd WITH stat
store_stages = {'NREM' : {'week2': [], 'week5': []}, 
                'REM': {'week2': [], 'week5': []}, 
                'WAKE': {'week2': [], 'week5': []}}

for  stage, title in zip(events_id, ['NREM', 'REM', 'WAKE']):
    psd_week2 = store_psd['week2']
    event_week2 = store_event['week2']
    psd_week5 = store_psd['week5']
    event_week5 = store_event['week5']
    name_week2 = store_name['week2']
    name_week5 = store_name['week5']


    finder2 = [ np.where(event_week2[i] == events_id_map[stage])[0] for i in range(len(event_week2)) ]
    psd_week2 = [ psd_week2[i][finder2[i], :, :] for i in range(len(event_week2)) ]
    sbj_mask2 = [len(f) > 0 for f in finder2]    
    name_week2 = [ n for n, m in zip(name_week2, sbj_mask2) if m]
    finder5 = [ np.where(event_week5[i] == events_id_map[stage])[0] for i in range(len(event_week5)) ]
    psd_week5 = [ psd_week5[i][finder5[i], :, :] for i in range(len(event_week5)) ]
    sbj_mask5 = [len(f) > 0 for f in finder5]    
    name_week5 = [ n for n, m in zip(name_week5, sbj_mask5) if m]
  
    store =  {'week2':[], 'week5':[]}
    for time, color, psd, name in zip(['week2', 'week5'], colors, [psd_week2, psd_week5], [name_week2, name_week5]):
        #set_trace()
        boots = my_bootstraper(psd, ch_indices=None, repetions=1000, n=10)
        boots = np.nanmean(boots, 0) #mean over bootstrap samples
        freqs_mask = np.logical_and(freqs>=fmin, freqs<=fmax) #mask to select bins
        boots = boots[:, :, freqs_mask].mean(2)  #sum power for selected bins
        #set_trace()
        #store[time].append(boots)
        store_stages[title][time] = boots
    
   
for stage_to_plot in ['NREM', 'REM', 'WAKE']:
    n2 = store_stages[stage_to_plot]['week2']
    n2  = n2[~np.isnan(n2)].reshape([-1, 129, 1]) #drop nan
    n2 = n2.transpose([0,2,1])

    n5 = store_stages[stage_to_plot]['week5']
    n5  = n5[~np.isnan(n5)].reshape([-1, 129, 1]) #drop nan
    n5 = n5.transpose([0,2,1])

    contrast = n5.mean(0).transpose([1,0]) - n2.mean(0).transpose([1,0])
    tfr = mne.EvokedArray(contrast, info=info, tmin=0.)
    t, clust, pv, _ = mne.stats.spatio_temporal_cluster_test([n2, n5])

    clust = [c for c, p in zip(clust, pv) if p <= 0.08]
    mask = np.zeros((129,1), dtype=bool)
    for i_c, c in clust:
        mask[c] = True

    tfr.plot_topomap(ch_type='eeg',times=[0],  scalings=1,
                    time_format=None, cmap='Reds', vmin=0, vmax=6, title=None,
                    units='F-value',  mask=mask, #cbar_fmt='%0.1f',
                    size=3, time_unit='s', outlines='skirt',  contours=0, #extrapolate='head',
                    mask_params=dict(marker='o',
                    markerfacecolor='black', markeredgecolor='black',
                    markeredgewidth=2, linewidth=0, markersize=3))
    #plt.savefig('contr_{}_{}_{}.tiff'.format(stage_to_plot, fmin, fmax), dpi=300)                   
    break