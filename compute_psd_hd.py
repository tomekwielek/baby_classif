import mne
import os
import numpy as np
import pandas as pd
from config import stag_fname, mysave
from functional import map_stag_with_raw, my_rename_col, order_channels
import glob
from IPython.core.debugger import set_trace
from config import markers35, markers27
from pyentrp import entropy as ent

sheet = 'St_Pr_corrected_35min'
if '35' in sheet:
    markers = markers35
elif '27' in sheet:
    markers = markers27


mff_path = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\'
edf_path = 'H:\\BABY\\data\\'

fnames_edf_ = sorted(list(filter(lambda x: x.endswith('ref100.edf'), os.listdir(edf_path))))
fnames_edf = [i.split('.edf')[0] for i in fnames_edf_]
fnames_mff = sorted(list(filter(lambda x: x.endswith('.mff'), os.listdir(mff_path))))

ss = pd.read_excel(stag_fname, sheet_name=sheet)
ss = my_rename_col(ss)
ss = ss.drop(ss.columns[[0, 1, 2]], axis=1) #drop cols like 'Condition', 'Minuten', etc

if sheet == 'St_Pr_corrected_27min':
    bad = ['205_1_S', '206_1_S', '208_1_S', '213_1_S', '238_1_S', '239_1_S', '213_2_S']
elif sheet == 'St_Pr_corrected_35min':
    bad = ['104_1_S']


def load_raw_and_stag(idf_stag, file):
    stag = ss.filter(like = idf_stag)
    raw = mne.io.read_raw_egi(file[0], preload=True, include=markers)
    events = mne.find_events(raw, stim_channel='STI 014', initial_event=True)
    raw, events = raw.resample(125., events=events)
    raw.events = events
    sfreq = raw.info['sfreq']
    tmin = events[0][0] / sfreq
    gw_offset = 3. * 60. * sfreq #3min in ds
    #tmax = (events[-1][0] + 180000) / sfreq
    tmax = (events[-1][0] + gw_offset) / sfreq
    raw_last_ts = raw.times.shape[0] / sfreq
    if tmax > raw_last_ts:
        tmax = int(raw_last_ts)
        print('Short file')
    raw.crop(tmin, tmax)
    raw.pick_types(eeg=True)
    raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)
    raw.filter(l_freq=1., h_freq=30., fir_design='firwin')

    return raw, stag

def encode(stag):
    stages_coding = {'N' : 1, 'R' : 2, 'W' : 3,'X' : 4, 'XW' : 5, 'WR' : 6, 'RW' : 7, 'XR' : 8, 'WN' : 9}
    stag['numeric'] = stag.replace(stages_coding, inplace=False).astype(float)
    return stag

def compute_psd_segm(raw, window=30):
    from mne.time_frequency import psd_welch
    from mne.io import RawArray

    def get_raw_array(data):
        ch_types  = ['eeg'] * chan_len
        info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types=ch_types)
        raw_array = RawArray(data, info=info)
        return raw_array

    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    store_psd = []
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        ra = get_raw_array(e_)
        psd, freqs = psd_welch(ra ,fmin=1,fmax=30)
        store_psd.append(psd)
    #psd array of shape [freqs, channs, epochs]
    return freqs, np.asarray(store_psd).transpose(2,1,0)

def compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=False):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    store_segm = []
    if mspe:
        scale = 5
        m = np.zeros((scale, chan_len, no_epochs)) # multiscale entropy
    else:
        m = np.zeros((chan_len, no_epochs)) # PE
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        print (e_.shape)
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        for j in range(chan_len):
            if mspe:
                m[:,j,i], _ = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)
            else:
                m[j,i] = ent.permutation_entropy(e_[j], order=embed, delay=tau)
        del e_
    return m



idfs = map_stag_with_raw(fnames_mff, ss, sufx='.mff')
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls

#select single subject
#sbj = '108_1_S'
#idfs = {k : v for k,v in idfs.items() if k == sbj}

for k, v in sorted(idfs.items()):
    k_short = k.split('_S')[0]
    pattern = mff_path + k_short + '*.mff'
    file =  glob.glob(pattern)

    if k not in bad:
        print (k)
        raw, stag = load_raw_and_stag(k, file)
        stag = encode(stag)
        #freqs, psd = compute_psd_segm(raw, window=30)
        #print('psd len is: {} and stag len is: {}'.format(psd.shape[-1], len(stag)))
        pe = compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=False)
        #mysave(var = [stag, psd, freqs], typ='psd_hd', sbj=k[:5])
        mysave(var = [stag, pe], typ='pe_hd', sbj=k[:5])
    elif k in bad:
        print ('Sbj dropped, see red annot. by H.L in excel file')
        continue
