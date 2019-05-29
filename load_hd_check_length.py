import mne
import os
import numpy as np
import pandas as pd
from config import stag_fname, mysave
from functional import map_stag_with_raw, my_rename_col, order_channels
import glob
from IPython.core.debugger import set_trace

mff_path = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\'
edf_path = 'H:\\BABY\\data\\'

fnames_edf_ = sorted(list(filter(lambda x: x.endswith('ref100.edf'), os.listdir(edf_path))))
fnames_edf = [i.split('.edf')[0] for i in fnames_edf_]
fnames_mff = sorted(list(filter(lambda x: x.endswith('.mff'), os.listdir(mff_path))))

ids_edf = [fnames_edf[i][:5] for i in range(len(fnames_edf))]
ids_mff = [fnames_mff[i][:5] for i in range(len(fnames_mff))]

match = sorted(list(set(ids_edf) & set(ids_mff)))
fnames_mff_match = list(np.asarray(fnames_mff)[[ ids_mff[i] in match for i in range(len(ids_mff)) ]])


markers = ['D221', 'D222', 'D223', 'D224', 'D225', 'DI92', 'D201', 'D202',
'D203', 'D204', 'D205', 'DI93', 'D121', 'D122', 'D123', 'D124', 'D125', 'DI91', 'D101',
'D102', 'D103', 'D104', 'D105', 'DI95', 'DI94']

for edf, mff in zip(fnames_edf, fnames_mff_match):

    raw_mff = mne.io.read_raw_egi(mff_path + mff, preload=True, include=markers)
    raw_edf =  mne.io.read_raw_edf(edf_path + edf+'.edf', preload=True)

    ev_mff = mne.find_events(raw_mff, stim_channel = 'STI 014')

    tmax = (ev_mff[-1][0] + 180000) / 1000
    mff_last_ts = raw_mff.times.shape[0] / 1000
    if tmax > mff_last_ts:
        print('Short file')
        print(mff)
        continue
    time_mff = (ev_mff[-1][0] - ev_mff[0][0]+ 180000) / 1000 // 60
    time_edf = raw_edf.times.shape[0] // raw_edf.info['sfreq'] // 60
    assert time_mff == time_edf
