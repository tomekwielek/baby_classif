import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        count_stag, vis_clf_probas, write_pickle, read_pickle, plot_confusion_matrix,\
                        plot_acc_depending_scale)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
#from classify_with_shuffling_v2 import classify_shuffle
from classify_with_shuffling_v2_old import classify_shuffle
from collections import defaultdict
from IPython.core.debugger import set_trace

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]
time = 5 # if 2 week2, elif 5 week5, elif 'cat' concatenate
n_folds = 2

#setup = 'psd'
setup = 'mspet1m3'
#iterate to compare statistically psd vs mspe based
#store_perfs = {'s1':[], 's2': [], 's3': [], 's4': []}
#store_perfs = {'mspet1m3' : [], 'psd' : []}

store_perfs = {'mspet1m3' : []}
#for setup in ['mspet1m3', 'psd']:
#for s in [1,2,3,4]: #taus
for i in range(2):
    s = 6 #taus no
    if setup == 'mspet1m3':
        # SIMPLE GENERALIZATION BASED ON MSPE
        mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
        mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
        mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
        mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

        mspe1_ = [ mspe1[i ][:s, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
        mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

        mspe2_ = [ mspe2[i ][:s, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
        mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

        if time == 2:
            data_pe, data_stag = mspe1, stag1
        elif time == 5:
            data_pe, data_stag = mspe2, stag2
        elif time == 'cat':
            data_pe = mspe1 + mspe2
            data_stag = stag1 + stag2

    elif setup == 'psd':
        # SIMPLE GENERALIZATION BASED ON PSD
        psd1, stag1, names1, freqs = load_single_append(path, fnames1, typ=setup)
        psd2, stag2, names2, freqs = load_single_append(path, fnames2, typ=setup)
        psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
        psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

        rel_psd1 = [psd1[i] / np.abs(np.sum(psd1[i], 0)) for i in range(len(psd1))]
        rel_psd1 = [ np.log10(rel_psd1[i]) for i in range(len(psd1)) ]
        rel_psd2 = [psd2[i] / np.abs(np.sum(psd2[i], 0)) for i in range(len(psd2))]
        rel_psd2 = [ np.log10(rel_psd2[i]) for i in range(len(psd2)) ]
        psd1 = rel_psd1
        psd2 = rel_psd2
        psd1 = [ psd1[i].reshape(-1, psd1[i].shape[-1]) for i in range(len(psd1)) ] # reshape
        psd2 = [ psd2[i].reshape(-1, psd2[i].shape[-1]) for i in range(len(psd2)) ] #reshape

        if time == 2:
            data_pe, data_stag = psd1, stag1
        elif time == 5:
            data_pe, data_stag = psd2, stag2
        elif time == 'cat':
            data_pe = psd1 + psd2
            data_stag = stag1 + stag2

    elif setup == 'psd_and_mspe':
        #Load mspe
        mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ='mspet1m3')
        mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ='mspet1m3')
        mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
        mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

        mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
        mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

        mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
        mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

        #load psd
        psd1, stag1, names1, freqs = load_single_append(path, fnames1, typ='psd')
        psd2, stag2, names2, freqs = load_single_append(path, fnames2, typ='psd')
        psd1, stag1 = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
        psd2, stag2 = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

        rel_psd1 = [psd1[i] / np.abs(np.sum(psd1[i], 0)) for i in range(len(psd1))]
        rel_psd1 = [ np.log10(rel_psd1[i]) for i in range(len(psd1)) ]
        rel_psd2 = [psd2[i] / np.abs(np.sum(psd2[i], 0)) for i in range(len(psd2))]
        rel_psd2 = [ np.log10(rel_psd2[i]) for i in range(len(psd2)) ]

        psd1 = rel_psd1
        psd2 = rel_psd2
        psd1 = [ psd1[i].reshape(-1, psd1[i].shape[-1]) for i in range(len(psd1)) ] # reshape
        psd2 = [ psd2[i].reshape(-1, psd2[i].shape[-1]) for i in range(len(psd2)) ] #reshape

        psd1_mspe1 = [ np.vstack([psd1[i], mspe1[i]]) for i in range(len(psd1)) ]
        psd2_mspe2 = [ np.vstack([psd2[i], mspe2[i]]) for i in range(len(psd2)) ]

        if time == 2:
            data_pe, data_stag = psd1_mspe1, stag1
        elif time == 5:
            data_pe, data_stag = psd2_mspe2, stag2
        elif time == 'cat':
            data_pe = psd1_mspe1 + psd1_mspe2
            data_stag = stag1 + stag2

    # Index for plot sbj
    #idx_plot = 1
    #assert names1[idx_plot].split('_')[0] == names2[idx_plot].split('_')[0]

    # Get all metrics
    perf  = classify_shuffle(data_pe, data_stag,  myshow=True, check_mspe=True, null=False,
                                n_folds=n_folds, search=False)

    #store_perfs['s'+str(s)].append(perf)

    #write_pickle(perf, 'mspe_perfs_cat.txt')
    #plot_acc_depending_scale(store_perfs)

    #Aggregate folds
    #accav = np.asarray([perf[i][0] for i in range(n_folds)]).mean(0)
    accav = np.asarray([perf[i][0] for i in range(len(perf))]).mean(0)
    print accav
    cmav = np.asarray([perf[i][1] for i in range(len(perf))]).mean(0)
    recall =  np.asarray([perf[i][2] for i in range(len(perf))]).mean(0)
    precission =  np.asarray([perf[i][3] for i in range(len(perf))]).mean(0)
    #f1perclass =  np.asarray([perf[i][4] for i in range(len(perf))]).mean(0)
    #cm_title = 'Confusion matrix'
    #plot_confusion_matrix(cmav, ['NREM', 'REM', 'WAKE'], title=cm_title, normalize=True)
    #store_perfs[setup].extend([perf])
    store_perfs[setup].extend((accav, recall, precission))









#Run shuffling
nulliter  = 1000
null_f1 = np.empty([nulliter])
null_f1_indiv = np.empty([nulliter, 3])
for idx in range(nulliter):
    f1_, f1_indiv_ = classify_shuffle(data_pe, data_stag, idx_plot, myshow=False, check_mspe=True, \
                                null=True, n_folds=n_folds, search=False)
    null_f1[idx] = f1_
    null_f1_indiv[idx] = f1_indiv_
    print(idx)


#write_pickle(store_acc, 'psd_vs_mspe_scores_cat.txt')
