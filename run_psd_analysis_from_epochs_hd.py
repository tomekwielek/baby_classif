'''
PSD analysis, statistics, visualisation
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
from functional import write_pickle, read_pickle

matplotlib.rcParams.update({'font.size': 12,'xtick.labelsize':8, 'ytick.labelsize':8})
np.random.seed(12)

path = 'H:\\BABY\\working\\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

events_id = ['N', 'R', 'W']
events_id_map = {'N':1, 'R':2, 'W':3}
events_id_map_rev = {v:k for k,v in events_id_map.items()}
store_psd =  {'week2': dict(zip(events_id, [[], [], []])),
            'week5': dict(zip(events_id, [[], [], []]))}
drop_artif = False
def find_drop_20hz(epoch):
    #Drop epochs with 20Hz artifacts (egi impedance check):
    #    - requiers 'bad subjects' to be defined (done by visual inspection of time-freq plots)
    #    - drop epochs with 20Hz power higher than 90th percentile
    psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128, n_overlap=64,
                                            picks=slice(0,6,1))
    freqs = freqs.astype(int)
    n_freqs = len(freqs)
    idx_freq = np.where(freqs == 20)
    band = psds[:,:,idx_freq].squeeze()
    band = band.mean((1))
    idx_time = np.where(band[:] > np.percentile(band[:], 90))[0]
    if sbj in ['236_2'] and 0 in idx_time:
        idx_time = idx_time[1:] #see annot by HL in excel and functional.py; BL shorter
    mask_psd = np.ones(psds.shape,dtype=bool)
    mask_psd[idx_time,:,:] = False
    return(epoch.drop(idx_time))

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        #empty folder means missing data, see exclusion criteria OR missalligned. PSD crashes with 212_2 (/)
        if len(os.listdir(os.path.dirname(paths(typ='epoch_hd', sbj=sbj)))) == 0 or sbj == '212_2':
            continue
        else:
            epoch = myload(typ='epoch_hd', sbj=sbj)
            epoch = epoch[events_id]
            this_events = np.unique(epoch.events[:,-1])
            if drop_artif == True and sbj in bad: #bad subject are defined by visual inspection of the tfr plots
                epoch_clean = find_drop_20hz(epoch)
            else:
                epoch_clean = epoch
            del epoch

            for ev in this_events:
                stage = events_id_map_rev[ev]
                this_epoch = epoch_clean.copy()[stage]
                psds, freqs =  mne.time_frequency.psd_welch(this_epoch,  fmin=12, fmax=15, n_fft=128,
                                                        picks='eeg', n_overlap=64)
                psds = 10. * np.log10(psds) #dB
                psds = psds.mean(0) #mean epochs
                freqs = freqs.astype(int)
                store_psd[time][stage].append(psds)

#Save psds
write_pickle(store_psd, 'psds_hd_from_epochs_12-14hz.txt')

store_psd = read_pickle('H:\\BABY\\results\\figs\\psd_final\\hd_topos\\psds_hd_from_epochs_12-14hz.txt')

'''
def count_nested(mydict):
    res = {'week2' : {}, 'week5':{}}
    for k, v in mydict.items():
        for kk, vv in mydict[k].items():
            count = np.asarray(vv).shape
            res[k][kk] = count
    return (res)
'''

def vmin_vmax_nested(mydict, stage):
    res = []
    for k, v in mydict.items():
        for kk, vv in mydict[k].items():
            if kk == stage:
                mymin = np.median(np.asarray(vv), 0).min() #av sbjs
                mymax = np.median(np.asarray(vv), 0).max()
                res.append((mymin, mymax))
    #set_trace()
    return (np.asarray(res)[:,0].max(), np.asarray(res)[:,1].min() )



 # PLOT TOPOMAPS
matplotlib.rcParams.update({'font.size': 16}) # ,'xtick.labelsize':8, 'ytick.labelsize':8})
for stag, title in zip(events_id, ['NREM', 'REM', 'WAKE']):
    store_tfr = []
    vmin , vmax = vmin_vmax_nested(store_psd, stage=stag)
    for time in ['week2', 'week5']:
        psd_data = store_psd[time][stag]
        psd_data = np.array(psd_data).mean(0)[:,:, np.newaxis]
        n_freq = psd_data.shape[1]
        #set_trace()
        tfr = mne.time_frequency.AverageTFR(info=epoch.info, data=psd_data, times=[0], freqs=np.arange(1,n_freq+1), nave=1)
        final_title = (' ').join([time, title])
        tfr.plot_topomap(title=final_title, fmin=None, fmax=None, size=3, cmap='Reds', vmin=vmin, vmax=vmax,
                    cbar_fmt='%3.f', res=100, contours=0)
        #plt.show()
        plt.tight_layout()

        plt.savefig('12-14hz_{}_{}'.format(time, stag))





#######################################
# PLOT DENSITIES (use for 1-30 psds)
colors = ['black', 'red']
freqs = np.arange(1,30)
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(6,3))

for ax, stage, title in zip([ax1, ax2, ax3],
                   ['N', 'R', 'W'], ['NREM', 'REM', 'WAKE']):
    psd_week2 = store_psd['week2'][stage]
    psd_week5 = store_psd['week5'][stage]

    store =  {'week2':[], 'week5':[]}
    for time, color, psd in zip(['week2', 'week5'], colors, [psd_week2, psd_week5]):
        av_psd = np.mean(psd, 1) #mean channels
        store[time].append(av_psd)
        av_psd = np.mean(av_psd, 0) #mean subjects

        #set_trace()
        ax.plot(freqs, av_psd, color=color, linewidth=3)
        ax.set(xlabel='Frequency [Hz]')

    av2 = store['week2'][0]
    av5 = store['week5'][0]
    count2 = len(av2)
    count5 = len(av5)
    sample = np.random.choice(range(max(count2, count5)), size=min(count2, count5), replace=False)
    av5 = av5[sample, :]
    X = av2 - av5
    t, clusters, pv, _ =  mne.stats.permutation_cluster_1samp_test(X, n_permutations=10000)
    print(pv)
    for i_c, c in enumerate(clusters):
        c = c[0]
        if pv[i_c] <= 0.05:
            h = ax.axvspan(freqs[c.start], freqs[c.stop - 1],
                            color='r', alpha=0.2)
            ax1.legend((h, ), ('cluster p-value < 0.05', ),  prop={'size': 6})
        else:
            ax.axvspan(freqs[c.start], freqs[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.2)

ax.set_xticks(np.arange(min(freqs), max(freqs)+1, 5))
#ax.set_xscale('log')
ax1.set(ylabel='Power Spectral Density [dB]', ylim=(-140, -95))
ax1.set_yticks(np.arange(-140, -95 , 10))
ax3.legend(['week2', 'week5'])
plt.tight_layout()
#plt.suptitle(' '.join(plot_chann))
plt.tight_layout()
plt.show()

plt.savefig('_'.join(plot_chann)+ 'psd.tif', dpi=300)
