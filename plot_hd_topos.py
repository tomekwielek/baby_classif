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

pick_chann = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4'] # set what channels to compute psd for 
plot_chann = ['F3', 'F4']
m = mne.channels.read_montage(kind='GSN-HydroCel-129')
epoch = myload(typ='epoch_hd', sbj='108_2')
info = epoch.info
del epoch

colors = ['black', 'red']
plot_lay = False #channels location plotting

def find_drop_20hz(epoch):
    '''
    Drop epochs with 20Hz artifacts (egi impedance check):
        - requiers 'bad subjects' to be defined (done by visual inspection of time-freq plots)
        - drop epochs with 20Hz power higher than 90th percentile
    '''
    psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128, n_overlap=64,
                                            picks=slice(0,6,1))
    freqs = freqs.astype(int)
    idx_freq = np.where(freqs == 20)
    band = psds[:,:,idx_freq].squeeze()
    band = band.mean((1))
    idx_time = np.where(band[:] > np.percentile(band[:], 90))[0]
    if sbj in ['236_2'] and 0 in idx_time:
        idx_time = idx_time[1:] #see annot by HL in excel and functional.py; BL shorter
    mask_psd = np.ones(psds.shape,dtype=bool)
    mask_psd[idx_time,:,:] = False
    return(epoch.drop(idx_time))


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

write_pickle((store_psd, store_event, store_name, freqs), 'hd_psds_stages_names_freqs.txt')
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
    #set_trace()
    return store_repet

# PLOT psd WITH stat
#ch_indices  = mne.pick_channels(epoch_clean.ch_names, include=plot_chann)
colors = ['black', 'red']
fmin,fmax = (8,14)
#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(6,3))
store_stages = {'NREM' : {'week2': [], 'week5': []}, 
                'REM': {'week2': [], 'week5': []}, 
                'WAKE': {'week2': [], 'week5': []}}

for ax, stage, title in zip([ax1, ax2, ax3],
                   events_id, ['NREM', 'REM', 'WAKE']):
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
        boots = my_bootstraper(psd, ch_indices=None, repetions=100, n=10)
        boots = np.nanmean(boots, 0) #mean over bootstrap samples
        freqs_mask = np.logical_and(freqs>=fmin, freqs<=fmax) #mask to select bins
        boots = boots[:, :, freqs_mask].mean(2)  #sum power for selected bins
        #set_trace()
        #store[time].append(boots)
        store_stages[title][time] = boots
    
n2 = store_stages['NREM']['week2']
n2  = n2[~np.isnan(n2)].reshape([-1, 129, 1]) #drop nan
n2 = n2.transpose([0,2,1])

n5 = store_stages['NREM']['week5']
n5  = n5[~np.isnan(n5)].reshape([-1, 129, 1]) #drop nan
n5 = n5.transpose([0,2,1])

tfr = mne.EvokedArray(contrast, info=info, tmin=0.)
t, clust, pv, _ = mne.stats.spatio_temporal_cluster_test([n2, n5])

clust = [c for c, p in zip(clust, pv) if p <= 0.07]
mask = np.zeros((129,1), dtype=bool)
for i_c, c in clust:
    mask[c] = True

#tfr.plot_topomap(cbar_fmt='%3.1f',  unit='', contours=0, vmin=np.min, vmax=np.max, mask=mask)
tfr.plot_topomap(ch_type='eeg',times=[0],  scalings=1,
                   time_format=None, cmap='Reds', vmin=np.min, vmax=np.max, title=None,
                   units='F-value', cbar_fmt='%0.1f', mask=mask,
                   size=3, time_unit='s', outlines='skirt', extrapolate='head', contours=0,
                   mask_params=dict(marker='o',
                   markerfacecolor='black', markeredgecolor='black',
                   markeredgewidth=2, linewidth=0, markersize=3))