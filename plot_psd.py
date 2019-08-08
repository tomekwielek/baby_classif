'''
PSD analysis, statistics, visualisation
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
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

pick_chann = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'] # set what channels mspe gets computed
plot_chann = ['O1', 'O2']
m = mne.channels.read_montage(kind='standard_1020')

colors = ['black', 'red']
plot_lay = False #channels location plotting

def find_drop_20hz(epoch):
    '''
    Drop epochs with 20Hz artifacts (egi impedance check):
        - requiers 'bad subjects' to be defined (done by visual inspection of time-freq plots)
        - drop epochs with 20Hz power higher than 90th percentile
    '''
    psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128, picks=slice(0,6,1))
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


# Plot layout 6 channels
if plot_lay:
    montage = mne.channels.read_montage(kind='standard_1020', ch_names=['F4', 'F3', 'C3', 'C4', 'O1', 'O2'])
    mne.viz.plot_montage(montage=montage, scale_factor=89, show_names=True)


for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        #empty folder means missing data, see exclusion criteria
        if len(os.listdir(os.path.dirname(paths(typ='epoch', sbj=sbj)))) == 0:
            continue
        else:
            epoch = myload(typ='epoch', sbj=sbj)
            epoch = epoch[events_id]
            epoch.set_montage(m)
            epoch.pick_channels(pick_chann)
            if sbj in bad: #bad subject are defined by visual inspection of the tfr plots
                epoch_clean = find_drop_20hz(epoch)
            else:
                epoch_clean = epoch

            del epoch
            psds, freqs = mne.time_frequency.psd_welch(epoch_clean, fmin=1, fmax=30, n_fft=128,
                                                    picks=pick_chann, n_overlap=64) # slice(0,6,1)
            psds = 10. * np.log10(psds) #dB
            freqs = freqs.astype(int)
            assert len(epoch_clean) == len(epoch_clean.events) == len(psds)
            store[time].append(epoch_clean)
            store_psd[time].append(psds)
            store_event[time].append(epoch_clean.events[:,-1])


def my_bootstraper(data, ch_indices, repetions=1000, n=10):
    '''
    Bootstraped averaging of n epochs (epochs per sleep stage)
    data[i].shape = n_epochs x n_chs x freqs where i=sbj index
    '''
    np.random.seed(None) # randomly initialize the RNG from some platform-dependent source
    store_repet = np.zeros([repetions, len(data), 29]) #29 freq bins
    for i in range(repetions):
        store = []
        for d_ in data:
            count = d_.shape[0]
            if count == 0:
                store.append([np.nan] * 29) #29 freq bins
            else:
                sample = np.min([n, count]) # if n<count mean over count
                epoch_indices = np.random.choice(count, sample)
                av = d_[:, ch_indices, :] #sample channels
                av = av[epoch_indices, :, :] #sample epochs
                store.append(av.mean((0,1)))
        #set_trace()
        store_repet[i,:] = store
    return store_repet

# PLOT psd WITH stat
ch_indices  = mne.pick_channels(epoch_clean.ch_names, include=plot_chann)
colors = ['black', 'red']
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(6,3))

for ax, stage, title in zip([ax1, ax2, ax3],
                   events_id, ['NREM', 'REM', 'WAKE']):
    psd_week2 = store_psd['week2']
    event_week2 = store_event['week2']
    psd_week5 = store_psd['week5']
    event_week5 = store_event['week5']

    finder2 = [ np.where(event_week2[i] == events_id_map[stage])[0] for i in range(len(event_week2)) ]
    psd_week2 = [ psd_week2[i][finder2[i], :, :] for i in range(len(event_week2)) ]
    finder5 = [ np.where(event_week5[i] == events_id_map[stage])[0] for i in range(len(event_week5)) ]
    psd_week5 = [ psd_week5[i][finder5[i], :, :] for i in range(len(event_week5)) ]

    store =  {'week2':[], 'week5':[]}
    for time, color, psd in zip(['week2', 'week5'], colors, [psd_week2, psd_week5]):
        boots = my_bootstraper(psd, ch_indices=ch_indices, repetions=4000, n=10)
        boots = np.nanmean(boots, 0) #mean over subjects bootstraped averages of epochs
        store[time].append(boots)
        av_psd = np.nanmean(boots, 0) #mean subjects
        ax.plot(freqs, av_psd, color=color, linewidth=3)
        ax.set(xlabel='Frequency [Hz]')

    av2 = store['week2'][0]
    av2  = av2[~np.isnan(av2)].reshape([-1, 29]) #drop nan
    av5 = store['week5'][0]
    av5  = av5[~np.isnan(av5)].reshape([-1, 29]) #drop nan
    count2 = len(av2)
    count5 = len(av5)
    sample = np.random.choice(range(max(count2, count5)), size=min(count2, count5), replace=False)
    av5 = av5[sample, :]
    X = av2 - av5
    t, clusters, pv, _ =  mne.stats.permutation_cluster_1samp_test(X, n_permutations=100000)
    print(pv)
    for i_c, c in enumerate(clusters):
        c = c[0]
        if pv[i_c] <= 0.05:
            h = ax.axvspan(freqs[c.start], freqs[c.stop - 1],
                            color='r', alpha=0.2)
            ax1.legend((h, ), ('cluster p-value < 0.05', ),  prop={'size': 6})
        else:
            ax.axvspan(freqs[c.start], freqs[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.2)

    ax.set_xticks(np.arange(min(freqs), max(freqs)+1, 5))
    #ax.set_xscale('log')
    ax1.set(ylabel='Power Spectral Density [dB]', ylim=(-140, -95))
    ax1.set_yticks(np.arange(-140, -95 , 10))
    ax3.legend(['week2', 'week5'])
plt.tight_layout()
plt.suptitle(' '.join(plot_chann))
plt.tight_layout()
plt.show()
plt.savefig('_'.join(plot_chann)+ 'psd.tif', dpi=300)
