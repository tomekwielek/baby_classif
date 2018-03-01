import mne
from mne import io
import os
import pandas as pd
import pickle
import numpy as np
from mne import EpochsArray
data_path = 'H:\\BABY\\data\\'

raw_fnames_ = sorted(np.array(filter(lambda x: x.endswith('.edf'), os.listdir(data_path))))
raw_fnames = [i.split('.edf')[0] for i in raw_fnames_] #dlete .edf from fname
stag_fname = 'H:\\BABY\\Stages_inklPrechtl_corrected.xlsx'
s = pd.read_excel(stag_fname, 'Staging vs_Prechtl_35min') #define sheet

def map_stag_with_raw(s): # get raw file name that corr. with stag name
    m = {s_ : [r_ for r_ in raw_fnames if np.logical_and(r_.startswith(s_[:5]), r_.endswith(s_[-6:]))] for s_ in s.columns}
    return m

def preproces(raw):
    raw.pick_types(eeg=True)
    raw.filter(l_freq=1, h_freq=30)
    return raw

def load_raw_and_stag(idf_stag, idf_raw):
    stag = s.filter(like = idf_stag)
    raw = io.read_raw_edf(data_path + idf_raw + '.edf', preload=True)
    return preproces(raw), stag

def get_epochs(raw, window=30):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    ch_types  = ['eeg'] *6 + ['ecg', 'eog', 'emg', 'eog']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    store_segm = []
    l = []
    for i in range(no_epochs):
        l.append(data[...,:window_dp])
        data = np.delete(data, np.s_[:window_dp], axis =1 )
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types = ch_types)
    epochs = EpochsArray(l, info=info)
    return epochs

def compute_welch_power(epochs):
    from mne.time_frequency import psd_welch
    psds, freqs = psd_welch(epochs, fmin=2,fmax=25, picks= range(10), n_fft=256)
    psds = 10 * np.log10(psds)
    psds = psds[...,::6]
    freqs = freqs[::6]
    return psds, freqs


idfs = map_stag_with_raw(s)
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty

store_sbs = []
for s_, r_ in idfs.items():
    print s_
    raw, stag = load_raw_and_stag(s_, r_[0])
    epochs = get_epochs(raw, window=30)
    power, freqs = compute_welch_power(epochs)
    l = [power, stag, freqs]
    store_sbs.append(l)


save_path = 'H:\\BABY\\working\\oscil\\'
def mysave(var, fname):
    save_name = save_path + fname
    with open(save_name, 'wb') as f:
        pickle.dump(var, f)
mysave(var = store_sbs, fname='power_stag_freqs.txt')
