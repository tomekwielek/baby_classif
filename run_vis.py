import mne
from mne import io
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

raw_path ='H:\\BABY\\data\\'
pe_path = 'H:\\BABY\\working\\'
power_path = 'H:\\BABY\\working\\oscil\\'
def myload(path, fname):
    load_name = path + fname
    with open(load_name, 'rb') as f:
        out = pickle.load(f)
    return out


pe = myload(pe_path, 'pe_all_tau1.txt')
stag = myload(pe_path, 'stag_all_m4.txt')
list_ = myload(power_path, 'power_stag_freqs.txt')
power = [list_[i][0]  for i in range(len(list_))]
freqs = list_[0][2]

ch_names = io.read_raw_edf(raw_path + '104_2_correctFilter_2heogs_ref100.edf',
        preload=False).info['ch_names'][:-1]

for i in [9]:#range(len(pe)):
    fig, axes = plt.subplots(3,1)
    #pe_ =
    df = pd.DataFrame(pe[i].transpose(), index=range(pe[i].shape[1]), columns=ch_names)
    df.plot(ax = axes[0])
    stag[i].plot(ax = axes[1])
    #axes[1].plot(stag[i].iloc[:,[1]].values)
    df_power = pd.DataFrame(power[i][..., 1], index=range(power[i].shape[0]), columns=ch_names)
    df_power.plot(ax = axes[2])
plt.show()
