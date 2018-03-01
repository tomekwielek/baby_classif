import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_stages_to_classif, cat_pe_stag, load_single_append,
                        vis_PCA, vis_ICA, classify_lso, classify_generalize_time)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
sel_idxs = [1,2,3]
print sel_idxs
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
f_ =paths(typ='raws') + '108_1_correctFilter_2heogs_ref100.edf'
ch_names = io.read_raw_edf(f_, preload=True).info['ch_names']
del f_

#fnames = [f for f in fnames if f.endswith('1') or f.endswith('2')] #filter folders
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

setups= ['pet1m3']

store_scores = dict()

for setup in setups:
    #read_raw(setup, sheet='St_Pr_corrected_35min')
    #pe1, stag1, _ = load_single_append(path, fnames1, typ = setup)
    #pe2, stag2, _ = load_single_append(path, fnames2, typ = setup)

    #pe1, stag1 = select_stages_to_classif(pe1, stag1, sel_idxs=sel_idxs)
    #pe2, stag2 = select_stages_to_classif(pe2, stag2, sel_idxs=sel_idxs)

    #report = classify_generalize_time(pe1, stag1, pe2, stag2, show=True, setup=setup)

    pe, stag, _ = load_single_append(path, fnames2, typ = setup)
    pe, stag = select_stages_to_classif(pe, stag, sel_idxs=sel_idxs)
    pred_pe, scores_pe, report, features = classify_lso(pe, stag, oversample=False,
                                show=False, setup=setup)
    print np.mean(scores_pe)
    #X, y = cat_pe_stag(pe, stag)
    #vis_PCA(X, y, sel_idxs)

    #feat_mean = np.asarray(features).mean(0)
    #feat_std = np.asarray(features).std(0)
    #feat_df = pd.DataFrame(zip(ch_names,feat_mean))
    #ax = feat_df.plot(kind="bar",rot=45,color="blue",fontsize=12, yerr=feat_std)
    #ax.set_xticklabels(ch_names[:10])
    #ax.set_ylabel('Channels importance')

    #pickle.dump(store_scores, open('perf_t2.txt', 'wb'))
#report.save('report_t1.html', overwrite=True)
