import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        vis_PCA, classify_lso, classify_generalize_time,
                        merge_stages, count_stag, subject_h0, group_h0, vis_clf_probas)
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
store_pval = dict()
for setup in setups:
    #read_raw(setup, sheet='St_Pr_corrected_35min')
    pe1, stag1, _ = load_single_append(path, fnames1, typ = setup)
    pe2, stag2, _ = load_single_append(path, fnames2, typ = setup)

    pe1, stag1 = select_class_to_classif(pe1, stag1, sel_idxs=sel_idxs)
    pe2, stag2 = select_class_to_classif(pe2, stag2, sel_idxs=sel_idxs)

    pred1, pred2, f1_1, f1_2, n1, n2  = classify_generalize_time(pe1, stag1, pe2, stag2, show=False, setup=setup)

    #pe, stag, _ = load_single_append(path, fnames2, typ = setup)
    #pe, stag = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)

    #pred_pe, scores_pe, _, f1s, names, f1s_each = classify_lso(pe, stag, show=False, setup=setup)
    #vis_clf_probas(pe, stag)

    #h0_gr, p  = group_h0(pe, stag,  f1s, times=30e4)

#temp
store = {}
h0_gr = np.asarray(h0_gr).mean(1)
store['scores']= scores_pe
store['pval'] = p
store['h0'] = h0_gr
import pickle
f = open('scores_t2_t1m3_fw_kfold2.txt', 'wb')
pickle.dump(store, f)


    for i in range(30):
        fig, ax = plt.subplots()
        sns.kdeplot(h0[:,i], ax = ax)
        ax.axvline(x=f1s[i], color='k', linestyle='--')
        #axes[i].plot(x=scores_pe[i], color='k', linestyle='--')
        fig.suptitle(names[i])
        print f1s[i]

    # plot group h0 and my av f1
    fig, ax = plt.subplots()
    sns.kdeplot(h0_gr_av, ax = ax)
    ax.axvline(x=my_av_f1, color='k', linestyle='--')
    ax.axvline(x=np.percentile(h0_gr_av, 99), color='r', linestyle='-')



    print np.asarray(f1s).mean(0)

    X, y = cat_pe_stag(pe, stag)
    vis_PCA(X, y, sel_idxs)

    #feat_mean = np.asarray(features).mean(0)
    #feat_std = np.asarray(features).std(0)
    #feat_df = pd.DataFrame(zip(ch_names,feat_mean))
    #ax = feat_df.plot(kind="bar",rot=45,color="blue",fontsize=12, yerr=feat_std)
    #ax.set_xticklabels(ch_names[:10])
    #ax.set_ylabel('Channels importance')

    #pickle.dump(store_scores, open('perf_t2.txt', 'wb'))
#report.save('report_t1.html', overwrite=True)
