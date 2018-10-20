import mne
from mne import io
import os
import pandas as pd
from pyentrp import entropy as ent
import numpy as np
from functional import map_stag_with_raw, my_rename_col, order_channels
from config import raw_path, stag_fname, mysave, pe_par
from IPython.core.debugger import set_trace
from functional import write_pickle, read_pickle
from collections import OrderedDict
import itertools
from sklearn.preprocessing import LabelEncoder

fnames_ = sorted(np.array(filter(lambda x: x.endswith('.edf'), os.listdir(raw_path))))
fnames = [i.split('.edf')[0] for i in fnames_] #delete .edf from fname

sheet = 'St_Pr_corrected_35min'
setup = 'mspet1m3'
sbj = '108_2_S'

s = pd.read_excel(stag_fname, sheet_name=sheet)
s = my_rename_col(s)
s = s.drop(s.columns[[0, 1, 2]], axis=1) #drop cols like 'Condition', 'Minuten', etc

# get stag-edf mapping
idfs = map_stag_with_raw(fnames, s, sufx='ref100')
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls
idfs = {k : v for k,v in idfs.items() if k == sbj}


#set pe parameters
if setup != 'psd':
    embed = pe_par[setup]['embed']
    tau = pe_par[setup]['tau']
#set typ_name for saving
if setup.startswith('pe'):
    typ_name = 'pet' + str(tau) + 'm' + str(embed)
elif setup.startswith('mspe'):
    typ_name = 'mspet' + str(tau) + 'm' + str(embed)
elif setup == 'psd':
    typ_name = setup
# set bad sbjs, defined by inspecting excell dat (e.g red annotations by Heidi)
if sheet == 'St_Pr_corrected_27min':
    # all but 213_2 considered as nicht auswertbar.
    # 213_2 schwierig + 'all physio channels flat, ECG reconstructed from EMG - pretty noisy'
    bad = ['205_1_S', '206_1_S', '208_1_S', '213_1_S', '238_1_S', '239_1_S', '213_2_S']

if sheet == 'St_Pr_corrected_35min':
    bad = ['104_1_S'] #104_1-S missing, pressumbly unscorable
else: bad = []

def preproces(raw):
    raw.pick_types(eeg=True)
    raw.filter(l_freq=1., h_freq=30., fir_design='firwin')
    return raw

def load_raw_and_stag(idf_stag, idf_raw):
    stag = s.filter(like = idf_stag)
    raw = io.read_raw_edf(raw_path + idf_raw + '.edf', preload=True, \
                        stim_channel=None)

    return preproces(raw), stag # apply pp.
    #return raw, stag #no pp.

def compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=False):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    if mspe:
        scale = 5
        m = np.zeros((scale, chan_len, no_epochs)) # multiscale entropy
    else:
        m = np.zeros((chan_len, no_epochs)) # PE
        srts = np.zeros((chan_len, no_epochs, window_dp - tau * (3 - 1), 3)) #patters
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        print (e_.shape)
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        for j in range(chan_len):
            if mspe:
                m[:,j,i] = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)
            else:
                m[j,i], srts[j, i, :] = ent.permutation_entropy(e_[j], m=embed, delay=tau)

        del e_
    return m, srts

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
        psd, freqs = psd_welch(ra ,fmin=1,fmax=30, n_overlap=128)
        psd = 10 * np.log10(psd) # or log is done latter on, in wrap_psd.py
        store_psd.append(psd)
    #psd array of shape [freqs, channs, epochs]
    return freqs, np.asarray(store_psd).transpose(2,1,0)

def encode(stag):
    stages_coding = {'N' : 1, 'R' : 2, 'W' : 3,'X' : 4, 'XW' : 5, 'WR' : 6, 'RW' : 7, 'XR' : 8, 'WN' : 9}
    stag['numeric'] = stag.replace(stages_coding, inplace=False).astype(float)
    return stag


for k, v in sorted(idfs.items()):

    if k not in bad:
        print (k)
        raw, stag = load_raw_and_stag(k, v[0])
        raw, _ = order_channels(raw) #reorder channels, add zeros for missings (EMG usually)
        stag = encode(stag)
        #pe, srts = compute_pe_segm(raw, embed=embed, tau=tau, mspe=False)

        freqs, psd = compute_psd_segm(raw, window=30)
        #mysave(var = [stag, psd, freqs], typ=typ_name, sbj=k[:5])
        #mysave(var = [stag, pe], typ=typ_name, sbj=k[:5])
    elif k in bad:
        print ('Sbj dropped, see red annot. by H.L in excel file')
        continue

# Select epoch and channel
def panda_patterns(data, epoch_idx, ch_idx):
    seq_patterns = []
    for i in range(srts.shape[2]):
        indexer = [all(patterns[k] == srts[ch_idx, epoch_idx, i, :]) for k,v in patterns.iteritems()]
        seq_patterns.append(patterns.keys()[np.where(indexer)[0][0]])

    le = LabelEncoder()
    df = pd.DataFrame(zip(srts[ch_idx, epoch_idx, ...], seq_patterns))
    df.rename(columns={0:'order',1:'pt'}, inplace = True)
    df['enc'] = le.fit_transform(df['pt'])
    plt.figure()
    df['enc'].plot.hist()
    return df

#Epoch raw signal
def get_epochs(raw, window=30):
    from mne import EpochsArray
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    store_epochs = []

    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        store_epochs.append(e_)
    epochs = EpochsArray(data=np.asarray(store_epochs),
                        info=raw.info,
                        events=None)
    return epochs


def plot_psd(psd, ax, ch_idx=1):
    psd = psd.transpose(0,2,1)
    times = range(psd.shape[1])
    #fig, ax = plt.subplots(figsize=(11,4))
    mesh = ax.pcolormesh(times, freqs, psd[:,:,ch_idx],
                         cmap='RdBu_r')

    ax.set(ylim=freqs[[0, -1]], xlabel='Epochs (30s)', ylabel='Frequency [Hz]')
    #cb = plt.colorbar(mesh)
    #cb.set_label('Log power')
    #cb.set_ticks([])
    #plt.title('Channel {}'.format(raw.ch_names[ch_idx]))
    #plt.show()



def plot_stag(stag, ax):
    #fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(stag['numeric'])
    ax.set(xlim=(0,len(stag)))
    #plt.show()



#epochs = get_epochs(raw)
#write_pickle((pe, srts), 'pe_patters.txt')
pe, srts = read_pickle('pe_patters.txt')

permutations = np.array(list(itertools.permutations(range(3))))
patterns = OrderedDict((('s' + str(i+1)), permutations[i])  for i in range(permutations.shape[0]))

ch_idx = 0
nrem_idx = 13
rem_idx = 37
wake_idx = 68
nrem_df = panda_patterns(srts, epoch_idx=nrem_idx, ch_idx=ch_idx)


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 1,
                       height_ratios=[4, 1]
                       )

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

plot_psd(psd,  ax=ax1, ch_idx=1)
plot_stag(stag, ax2)
