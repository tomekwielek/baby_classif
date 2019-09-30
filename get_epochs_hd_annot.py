'''
Get high density epochs with staging as annotations (events). Used only for hd_topoplots
'''
import mne
import os
import numpy as np
import pandas as pd
from config import stag_fname, mysave
from functional import map_stag_with_raw, my_rename_col, order_channels, find_drop_20hz
import glob
from IPython.core.debugger import set_trace
from config import markers35, markers27
from pyentrp import entropy as ent

sheet = 'St_Pr_corrected_35min' #Define

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

m = mne.channels.read_montage(kind='GSN-HydroCel-129')

events_id = ['N', 'R', 'W']

if sheet == 'St_Pr_corrected_27min':
    bad = ['205_1_S', '206_1_S', '208_1_S', '213_1_S', '238_1_S', '239_1_S', '213_2_S']
elif sheet == 'St_Pr_corrected_35min':
    bad = ['104_1_S']


def get_epochs_use_annotations(idf_stag, file):
    stag = ss.filter(like=idf_stag)
    if k.startswith('110_2'): #see annot by HL in excel; BL shorter
        stag = stag.iloc[:66]
    if k.startswith('236_2'):
        stag  = stag[1:] #see annot by HL in excel; BL shorter
    stag = stag.dropna()
    raw = mne.io.read_raw_egi(file[0], preload=True, include=markers)
    events = mne.find_events(raw, stim_channel='STI 014', initial_event=True)
    raw, events = raw.resample(125., events=events)
    raw.events = events
    sfreq = raw.info['sfreq']
    tmin = events[0][0] / sfreq
    gw_offset = 3. * 60. * sfreq #3min in ds
    tmax = (events[-1][0] + gw_offset) / sfreq
    raw_last_ts = raw.times.shape[0] / sfreq
    if tmax > raw_last_ts:
        tmax = int(raw_last_ts)
        print('Short file')
    raw.crop(tmin, tmax)
    raw.pick_types(eeg=True)
    raw, _ = mne.set_eeg_reference(raw, 'average', projection=True)
    raw.apply_proj()
    raw.filter(l_freq=1., h_freq=30., fir_design='firwin')

    n_epochs = np.floor(raw.times[-1] / 30).astype(int)
    if len(stag) - n_epochs == 1:# can be shorter due to windowing
        stag = stag[:-1]
    raw.info['meas_date'] = 0
    onset = np.arange(0, raw.times[-1], 30 )[:-1]
    duration = [30] * n_epochs
    description = stag[k].values
    if  all([len(duration) == len(onset) == len(description)]): #check for allignment
        annotations = mne.Annotations(onset, duration, description, orig_time=0)
        raw.set_annotations(annotations)
        event_id  = {'N' : 1,
                    'R' : 2,
                    'W' : 3,
                    'X' : 4,
                    'XW' : 5,
                    'WR' : 6,
                    'RW' : 7,
                    'XR' : 8,
                    'WN' : 9}
        events, event_id_this = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)
        raw.info['meas_date'] = None
        tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

        epochs = mne.Epochs(raw=raw, events=events,
                          event_id=event_id_this, tmin=0., tmax=tmax, baseline=None)

        return epochs

idfs = map_stag_with_raw(fnames_mff, ss, sufx='.mff')
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls

#idfs = {i : v for i, v in idfs.items() if i == '204_1_S'} # single subject

for k, v in sorted(idfs.items()):
    k_short = k.split('_S')[0]
    pattern = mff_path + k_short + '*.mff'
    file =  glob.glob(pattern)

    if k not in bad:
        print (k)
        epoch = get_epochs_use_annotations(k, file)
        if epoch is None: #
            print('NO alignement between stag and raw, ignore subject {}'.format(k))
            continue
        epoch = epoch[events_id]
        epoch.set_montage(m)

        if k in bad: #bad subject are defined by visual inspection of the tfr plots
            epoch_clean = find_drop_20hz(epoch)
        else:
            epoch_clean = epoch

        del epoch
        mysave(var=epoch_clean, typ='epoch_hd', sbj=k[:5])
