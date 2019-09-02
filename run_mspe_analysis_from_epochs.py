'''
MSPE analysis, statistics, visualisation
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2, mysave
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
from pyentrp import entropy as ent

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

pick_chann = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'] # select channels

m = mne.channels.read_montage(kind='standard_1020')
colors = ['black', 'red']

#plot layout 6 channesl
#montage = mne.channels.read_montage(kind='standard_1020', ch_names=['F4', 'F3', 'C3', 'C4', 'O1', 'O2'])
#mne.viz.plot_montage(montage=montage, scale_factor=89, show_names=True)

def find_drop_20hz(epoch): #drop epochs with 20Hz artifacts (egi impedance check)
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

def compute_mspe_from_epochs(epoch, embed=3, tau=1, window=30, mspe=True):
    data = epoch.get_data()
    sfreq = epoch.info['sfreq']
    #set_trace()
    no_epochs, no_chs, _ = data.shape
    if mspe:
        scale = 5
        m = np.zeros((scale, no_chs, no_epochs)) # multiscale permutation entropy
    else:
        m = np.zeros((no_chs, no_epochs)) # permutation entropy
    for i in range(no_epochs):
        for j in range(no_chs):
            if mspe:
                m[:,j,i], _ = ent.multiscale_permutation_entropy(data[i,j,:], m=3, delay=1, scale=scale)
            else:
                m[j,i] = ent.permutation_entropy(data[i,j,:], m=embed, delay=tau)
    return m

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        print(sbj)
        if len(os.listdir(os.path.dirname(paths(typ='epoch', sbj=sbj)))) == 0: #emty folder means bad sbj
            continue
        else:
            epoch = myload(typ='epoch', sbj=sbj)
            epoch = epoch[events_id]
            epoch.set_channel_types({'F3': 'eeg',
                                    'F4': 'eeg',
                                    'C3': 'eeg',
                                    'C4': 'eeg',
                                    'O1': 'eeg',
                                    'O2': 'eeg',
                                    'ECG':'ecg',
                                    'EMG':'emg',
                                    'VEOG':'eog',
                                    'HEOG_l':'eog',
                                    'HEOG_r':'eog'})
            epoch = epoch.filter(l_freq=1, h_freq=30, picks='eeg')
            epoch.set_montage(m)
            epoch.pick_channels(pick_chann)
            if sbj in bad: #bad subject are defined by visual inspection of the tfr plots
                epoch_clean = find_drop_20hz(epoch)
            else:
                epoch_clean = epoch

            del epoch
            mspe = compute_mspe_from_epochs(epoch_clean)
            stag = epoch_clean.events[:,-1]
            assert len(epoch_clean) == len(epoch_clean.events) == mspe.shape[-1]
            mysave(var = [stag, mspe], typ='mspe_from_epochs', sbj=sbj)
