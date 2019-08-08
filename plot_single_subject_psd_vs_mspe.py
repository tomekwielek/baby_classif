import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
from pyentrp import entropy as ent
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

matplotlib.rcParams.update({'font.size': 12,'xtick.labelsize':8, 'ytick.labelsize':8})
np.random.seed(12)
events_id_map = {'NREM':1, 'REM':2, 'WAKE':3}
events_id = ['N', 'R', 'W']
sbj = '110_2'
pick_chann = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
scale = 4
ch_idx  = 4

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

epoch = find_drop_20hz(epoch)
mspe = compute_mspe_from_epochs(epoch)
mspe = mspe[scale, :, :]
stag = epoch.events[:,-1]

psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128,
                                        picks=pick_chann, n_overlap=64) # slice(0,6,1)

mydict = {'mspe': {'NREM' :[], 'REM' : [], 'WAKE': []}, 'psds':  {'NREM' :[], 'REM' : [], 'WAKE': []}}

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(6,3))
for ax, this_stage, title in zip([ax1, ax2, ax3],
                   [1, 2, 3], ['NREM', 'REM', 'WAKE']):
    finder = np.where(stag == this_stage)[0]
    mydict['mspe'][title] = mspe[ch_idx, finder]

def plot_psd(psd, ax, ch_idx, fmin=1, fmax=30):
    psd = psd.transpose(2,0,1)
    times = range(psd.shape[1])

    freq_mask = np.logical_and(freqs>=fmin, freqs<=fmax)
    freqs_masked = freqs[freq_mask]
    mesh = ax.pcolormesh(times, freqs_masked, psd[fmin:fmax,:,ch_idx],
                         cmap='coolwarm')

    ax.set(ylim=freqs_masked[[0, -1]], ylabel='Frequency [Hz]', xticks=[])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cb = plt.colorbar(mesh, cax=cax, orientation='vertical')
    cb.set_label('log power')
    cb.set_ticks([])
    ax.set_title('{}'.format(epoch.ch_names[ch_idx]))

def plot_stag(stag, ax):
    times = range(len(stag))
    #multicolourded line
    for i, (x, y) in enumerate(zip(times, stag)):
        print(x)
        print(y)
        if y == 1:
            ax.hlines(y, x, x+1, 'darkblue', linewidth=4)
        elif y == 2:
            ax.hlines(y, x, x+1, 'red', linewidth=4)
        elif y == 3:
            ax.hlines(y, x, x+1,'gold', linewidth=4)
    unique_cl = np.unique(stag)
    unique_key = [[k for k, v in events_id_map.items() if v == s ][0] for s in unique_cl ]
    ax.set(xlim=(0,len(stag)-1), ylim=(0.7, 3.3), yticks=unique_cl, yticklabels=unique_key, xlabel='Epochs (30s)',
            ylabel='Sleep stage')

gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.06)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
pos2 = ax2.get_position()
pos2_ch = [pos2.x0 , pos2.y0 ,  pos2.width- 0.02, pos2.height]
ax2.set_position(pos2_ch)

psds = 10 * np.log10(psds)
plot_psd(psds,  ax=ax1, ch_idx=ch_idx)
plot_stag(stag, ax2)
plt.show()
#plt.savefig('single_subject_psd.tif', dpi=300)

# Plot mspe as box plots
matplotlib.rcParams.update({'font.size': 16,'xtick.labelsize':20, 'ytick.labelsize':14})
df = pd.DataFrame.from_dict(mydict['mspe'], orient='index')
df = df.T.reset_index()
df = df.melt(value_vars= ['NREM', 'REM', 'WAKE'] )
df = df.dropna()

fig, ax  = plt.subplots(figsize=[4,3.5])
my_pal = {'NREM': 'darkblue', 'REM': 'red', 'WAKE':'gold'}
box = sns.boxplot(x='variable', y='value', data=df, linewidth=2, palette=my_pal,\
                 ax=ax, whis=[5, 95], showfliers=False, dodge=True, order=['NREM', 'REM', 'WAKE'])
sns.swarmplot(x='variable', y='value', data=df, split=True, color='black', \
            size=4, alpha=0.7, ax=ax, order=['NREM', 'REM', 'WAKE'])
if scale == 0:
    title = 'Original signal (C4)'
else:
    title = 'Coarse-grained signal (C4)'
ax.set( ylabel = 'MSPE(scale={}) [bit]'.format(scale+1), title=title, xlabel='')
#ylim=(1.25, 1.55),
ticks = ax.get_yticks()[::2]
ax.set_yticks(ticks)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .5))
plt.tight_layout()
plt.savefig('scale{}'.format(scale+1) + 'single_subject_mspe.tif', dpi=300)
