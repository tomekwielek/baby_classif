import mne
from mne import io
import os
import pandas as pd
from pyentrp import entropy as ent
import numpy as np
from functional import map_stag_with_raw, my_rename_col, order_channels
from config import raw_path, stag_fname, mysave, pe_par
from IPython.core.debugger import set_trace
from functional import write_pickle, read_pickle, select_class_to_classif
from collections import OrderedDict
import itertools
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from https://www.elenacuoco.com/2016/07/31/simulating-time-series/
#ARMA
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from pyentrp import entropy as ent


fnames_ = sorted(list(filter(lambda x: x.endswith('.edf'), os.listdir(raw_path))))
fnames = [i.split('.edf')[0] for i in fnames_] #delete .edf from fname

sheet = 'St_Pr_corrected_35min'
setup = 'mspet1m3'
sbj = '110_2_S' # manuscript intern revied verison
#sbj = '118_1_S'


stages_coding = {'N' : 1, 'R' : 2, 'W' : 3,'X' : 4, 'XW' : 5, 'WR' : 6, 'RW' : 7, 'XR' : 8, 'WN' : 9}

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
        scale = 6
        m = np.zeros((scale, chan_len, no_epochs)) # multiscale entropy
    else:
        m = np.zeros((chan_len, no_epochs)) # PE
        srts = np.zeros((chan_len, no_epochs, window_dp - tau * (3 - 1), 3)) #patters

    store_eps = [] #store srts epochs wise

    for i in range(no_epochs):
        store_chs = [] #store srts channels wise
        e_ = data[...,:window_dp]
        print (e_.shape)
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        for j in range(chan_len):
            if mspe:
                #m[:,j,i], srts = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)
                m[:,j,i] = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)
                #store_chs.append(srts)

            else:
                #m[j,i], srts[j, i, :] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
                m[j,i] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
        store_eps.append(store_chs)

        del e_
    return m

def relative_power(data, sf, band, window_sec=1.):
    # Adapted from https://raphaelvallat.com/bandpower.html
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 2d-array
        Input signal in the chs x time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2

    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band
    data = data * 10e5
    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    rel_psd = np.zeros((11, 30)) #empty array define #freqs bins !!

    for chi in range(data.shape[0]): #iterate channels
        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data[chi, :], sf, nperseg=nperseg)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        #bp = simps(psd[idx_band], dx=freq_res)
        #  OR simple sum over bins, Relative power see: Xiao(2018) 'Electroencephalography power and coherence changes with age and motor skill'
        bp = psd[idx_band].sum()
        #rel_psd[chi, :] = psd[idx_band] / bp
        rel_psd[chi, :] = psd[idx_band] / bp
    return rel_psd, freqs[idx_band]

def compute_psd_segm(raw, window=30):
    def get_raw_array(data):
        from mne.io import RawArray
        ch_types  = ['eeg'] * chan_len
        info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types=ch_types)
        raw_array = RawArray(data*10e5, info=info) #convert to volts
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
        #ra = get_raw_array(e_) #for welch by mne
        #psd, freqs = psd_welch(ra ,fmin=1,fmax=30, n_per_seg=4*sfreq)
        psd, freqs = relative_power(e_, sfreq, [1.,30.])
        store_psd.append(psd)
    #psd array of shape [freqs, channs, epochs]
    return freqs, np.asarray(store_psd).transpose(2,1,0)

def encode(stag):
    stag['numeric'] = stag.replace(stages_coding, inplace=False).astype(float)
    return stag

sel_idxs = [1,2,3]
for k, v in sorted(idfs.items()):
    if k not in bad:
        print (k)
        raw, stag = load_raw_and_stag(k, v[0])
        raw, _ = order_channels(raw) #reorder channels, add zeros for missings (EMG usually)
        stag = encode(stag)

        #pe, srts = compute_pe_segm(raw, embed=embed, tau=tau, mspe=True)
        pe = compute_pe_segm(raw, embed=embed, tau=tau, mspe=True)

        if sbj == '110_2_S': #see annot by HL in excel; BL shorter
            stag = stag.iloc[:66]
        if len(stag) - pe.shape[-1] == 1:#pe can be shorter due to windowing
            stag = stag[:-1]
        pe, stag, idx_selected = select_class_to_classif([pe], [stag], sel_idxs)
        pe = pe[0]
        stag =stag[0]
        #srts = [srts[i] for i in range(len(srts)) if i in idx_selected]

        freqs, psd = compute_psd_segm(raw, window=30)
        psd = psd[:,:, idx_selected]

    elif k in bad:
        print ('Sbj dropped, see red annot. by H.L in excel file')
        continue

# SAVE
#write_pickle((pe, srts, psd, freqs), '110_2_S_psd_mspe_data_plots.txt')
#pe, srts, psd = read_pickle('110_2_S_psd_mspe_data_plots.txt')

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

def panda_patterns_plot(data, pe, epoch_idx, ch_idx, ax):
    '''
     PE CHECK!!!!!!!!!!!! srts data variables
    '''
    seq_patterns = []
    for i in range(data.shape[2]):
        indexer = [all(patterns[k] == data[ch_idx, epoch_idx, i, :]) for k,v in patterns.iteritems()]
        seq_patterns.append(patterns.keys()[np.where(indexer)[0][0]])

    le = LabelEncoder()
    df = pd.DataFrame(zip(data[ch_idx, epoch_idx, ...], seq_patterns))
    df.rename(columns={0:'order',1:'pt'}, inplace = True)
    df['enc'] = le.fit_transform(df['pt'])
    ax.hist(df['enc'])
    ax.text(0.1, 3, 'PE={}'.format(pe[ch_idx, epoch_idx]))
    return df

def panda_patterns_plot_mspe(data, mspe, epoch_idx, ch_idx, ax, scale):
    '''
    FOR MSPE
    '''
    data = data[epoch_idx][ch_idx][scale]

    seq_patterns = []
    for i in range(data.shape[0]):
        indexer = [all(patterns[k] == data[i, :]) for k,v in patterns.iteritems()]
        seq_patterns.append(patterns.keys()[np.where(indexer)[0][0]])

    le = LabelEncoder()
    df = pd.DataFrame(zip(data, seq_patterns))
    #set_trace()
    df.rename(columns={0:'order',1:'pt'}, inplace = True)
    df['enc'] = le.fit_transform(df['pt'])
    h_weights = np.ones_like(df['enc'])/float(len(df['enc']))
    ax.hist(df['enc'], bins=11, weights=h_weights)
    ax.set(ylim=(0, 0.5), xticks=[0,1,2,3,4,5])
    ax.set_xticklabels(labels = patterns.keys(), fontsize=16)
    ax.text(0.1, 0.45, 'PE={:4.4}'.format(mspe[scale, ch_idx, epoch_idx]), fontsize=16)
    return None

def plot_psd(psd, ax, ch_idx, fmin=1, fmax=30): # Define what freqs to plot):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    psd = psd.transpose(0,2,1)
    times = range(psd.shape[1])

    freq_mask = np.logical_and(freqs>=fmin, freqs<=fmax)
    freqs_masked = freqs[freq_mask]
    #set_trace()
    mesh = ax.pcolormesh(times, freqs_masked, psd[fmin:fmax,:,ch_idx],
                         cmap='viridis')

    ax.set(ylim=freqs_masked[[0, -1]], ylabel='Frequency [Hz]', xticks=[])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cb = plt.colorbar(mesh, cax=cax, orientation='vertical')
    cb.set_label('Realtive power')
    cb.set_ticks([])
    ax.set_title('Channel {}'.format(raw.ch_names[ch_idx]))
    ax.axvline(times[nrem_idx], color='black', alpha=0.9, linestyle='--')
    ax.axvline(times[rem_idx], color='black', alpha=0.9, linestyle='--')
    ax.axvline(times[wake_idx], color='black', alpha=0.9, linestyle='--')


def plot_stag(stag, ax):
    times = range(len(stag))
    ax.plot(stag['numeric'])
    unique_cl = np.unique(stag['numeric'])
    unique_key = [[k for k, v in stages_coding.items() if v == s ][0] for s in unique_cl ]
    ax.set(xlim=(0,len(stag)-1), yticks=unique_cl, yticklabels=unique_key, xlabel='Epochs (30s)',
            ylabel='Sleep stage')
    ax.axvline(times[nrem_idx], color='black', alpha=0.9, linestyle='--')
    ax.axvline(times[rem_idx], color='black', alpha=0.9, linestyle='--')
    ax.axvline(times[wake_idx], color='black', alpha=0.9, linestyle='--')

#pe, srts = read_pickle('pe_patters.txt')
permutations = np.array(list(itertools.permutations(range(3))))
patterns = OrderedDict((('s' + str(i+1)), permutations[i])  for i in range(permutations.shape[0]))

#PSD vs staging
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.06)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
pos2 = ax2.get_position()
pos2_ch = [pos2.x0 , pos2.y0 ,  pos2.width- 0.02, pos2.height]
ax2.set_position(pos2_ch)


# Define channls, scale and epochs idx to plot histograms for
ch_idx = 0 # 4 = C4
scale = 4
nrem_idx = 20
rem_idx = 3
wake_idx = 60

plot_psd(psd,  ax=ax1, ch_idx=ch_idx)
plot_stag(stag, ax2)
plt.show()

'''
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(7,4))
panda_patterns_plot_mspe(srts, pe,  epoch_idx=nrem_idx, ch_idx=ch_idx, ax=ax1, scale=scale)
ax1.set_ylabel('Normalized frequency', size = 15)
panda_patterns_plot_mspe(srts, pe, epoch_idx=rem_idx, ch_idx=ch_idx, ax=ax2, scale=scale)
panda_patterns_plot_mspe(srts, pe, epoch_idx=wake_idx, ch_idx=ch_idx, ax=ax3, scale=scale)
#plt.suptitle('Scale={}'.format(str(scale)))
'''

'''
def single_patts_plots():
    patts = list(itertools.permutations([0,1,2]))
    for i, p in enumerate(patts):
        plt.figure(figsize=(1,1))
        plt.plot([0,1,2], p , linestyle='-', marker='o', color='black')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig('p'+str(i)+'.tiff')
single_patts_plots()
'''
#PE histograms
'''
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
panda_patterns_plot(srts, pe,  epoch_idx=nrem_idx, ch_idx=ch_idx, ax=ax1)
panda_patterns_plot(srts, pe, epoch_idx=rem_idx, ch_idx=ch_idx, ax=ax2)
panda_patterns_plot(srts, pe, epoch_idx=wake_idx, ch_idx=ch_idx, ax=ax3)
plt.suptitle('fds')
'''
