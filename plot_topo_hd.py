import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        count_stag, vis_clf_probas, write_pickle, read_pickle, plot_confusion_matrix,\
                        plot_acc_depending_scale, remove_20hz_artif, get_unique_name_list)
from matplotlib import pyplot as plt
from mne import io
import pandas as pd
import pickle
from IPython.core.debugger import set_trace
from config import bad_sbjs_1, bad_sbjs_2
import mne
#from plot_pe_psd_stag import plot_data

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]

f = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\108_2_20150202.mff'
stag, psd, freqs = myload(typ='psd_hd', sbj='108_2')
#psd = psd.mean(-1)
psd = psd[:,:-1] #drop last chs, empty
psd = psd * 10e6
psd = psd / np.abs(np.sum(psd, 0))
psd =  np.log10(psd)
#psd_t = psd[:,:, np.newaxis].transpose([1,0,2]) #when average wpochs
psd_t = psd.transpose([1,0,2]) #no av


markers = ['D221', 'D222', 'D223', 'D224', 'D225', 'DI92', 'D201', 'D202',
'D203', 'D204', 'D205', 'DI93', 'D121', 'D122', 'D123', 'D124', 'D125', 'DI91', 'D101',
'D102', 'D103', 'D104', 'D105', 'DI95', 'DI94']
m = mne.channels.read_montage('GSN-HydroCel-129')
raw = mne.io.read_raw_egi(f, preload=True, include=markers).pick_types(eeg=True)
raw.ch_names.pop()
info = mne.create_info(ch_names = raw.ch_names, sfreq=125., ch_types='eeg', montage=m)
times = range(psd.shape[-1])
ep = mne.time_frequency.AverageTFR(info=info, data=psd_t, times=times, freqs=freqs, nave=1)

ep.plot_topomap(fmin=1, fmax=2, tmin=12, tmax=44)


psd = psd[:,:,np.newaxis]
psd_t = psd.transpose([1,0,2])
ep = mne.time_frequency.AverageTFR(info=info, data=psd_t, times=[1], freqs=freqs, nave=1)
ep.plot_topomap(scale=1)
