'''
MSPE analysis, statistics, visualisation
'''
import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
from pyentrp import entropy as ent
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
store_mspe =  {'week2': dict(zip(events_id, [[], [], []])),
            'week5': dict(zip(events_id, [[], [], []]))}


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
        print('Epoch no {}'.format(i))
        for j in range(no_chs):
            print('Chan no {}'.format(j))
            if mspe:
                m[:,j,i], _ = ent.multiscale_permutation_entropy(data[i,j,:], m=3, delay=1, scale=scale)
            else:
                m[j,i] = ent.permutation_entropy(data[i,j,:], m=embed, delay=tau)
    return m

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        print('Sbj {}'.format(sbj))
        #empty folder means missing data, see exclusion criteria OR missalligned
        if len(os.listdir(os.path.dirname(paths(typ='epoch_hd', sbj=sbj)))) == 0:
            continue
        else:
            epoch = myload(typ='epoch_hd', sbj=sbj)
            epoch = epoch[events_id]
            this_events = np.unique(epoch.events[:,-1])
            for ev in this_events:
                stage = events_id_map_rev[ev]
                this_epoch = epoch.copy()[stage]
                mspe = compute_mspe_from_epochs(this_epoch, embed=3, tau=1, window=30, mspe=True)
                #set_trace()
                store_mspe[time][stage].append(mspe)

write_pickle(store_mspe, 'mspe_hs_from_epochs.txt')
store_mspe = read_pickle('H:\\BABY\\results\\figs\\mspe_final\\mspe_hd_from_epochs.txt')

def vmin_vmax_nested(mydict, stage):
    res = []
    for k, v in mydict.items():
        for kk, vv in mydict[k].items():
            if kk == stage:
                vv = [ vv[i][scale, :, :] for i in range(len(vv)) ]
                #set_trace()
                mymin = np.median(np.hstack(vv), 1).min() #av sbjs
                mymax = np.median(np.hstack(vv), 1).max()
                #mymin = np.percentile(np.median(np.hstack(mspe_data), 1), 5) #av sbjs
                #mymax = np.percentile(np.median(np.hstack(mspe_data), 1), 95)
                res.append((mymin, mymax))
    return (np.asarray(res)[:,0].max(), np.asarray(res)[:,1].min() )

 # PLOT TOPOMAPS for MSPE
scale = 1
for stag in events_id:
    store_tfr = []
    vmin , vmax = vmin_vmax_nested(store_mspe, stage=stag)
    for time in ['week2', 'week5']:
        mspe_data = store_mspe[time][stag]
        mspe_data = [mspe_data[i][scale, :, :] for i in range(len(mspe_data)) ]
        mspe_data = np.hstack(mspe_data).mean(1)[:,np.newaxis, np.newaxis] # av epochs
        #mspe_data = np.array(mspe_data).mean(0)[:,np.newaxis, np.newaxis]
        #set_trace()
        tfr = mne.time_frequency.AverageTFR(info=epoch.info, data=mspe_data, times=[0], freqs=[0], nave=1)
        title = (' ').join([time, stag])
        #vmin, vmax = col_bounds[stag]
        tfr.plot_topomap(title=title, fmin=None, fmax=None, size=3, cmap='Reds', vmin=vmin, vmax=vmax,
                    cbar_fmt='%3.1f', res=300, contours=0)
        #plt.show()
        plt.tight_layout()
        plt.savefig('scale{}_{}{}'.format(scale+1, time, stag))
