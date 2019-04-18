import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        count_stag, vis_clf_probas, write_pickle, read_pickle, plot_confusion_matrix,\
                        plot_acc_depending_scale, remove_20hz_artif, get_unique_name_list)
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
from config import bad_sbjs_1, bad_sbjs_2
#from plot_pe_psd_stag import plot_data

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
#store_perfs = {'mspet1m3' : []}
store_perfs = []
#for setup in ['mspet1m3', 'psd']:
#for s in [1,2,3,4]: #taus
#for i in range(2):
s = 5 #taus no

mspe1, mspe_stag1, mspe_names1, _ = load_single_append(path, fnames1, typ='mspet1m3')
mspe2, mspe_stag2, mspe_names2, _ = load_single_append(path, fnames2, typ='mspet1m3')
psd1, psd_stag1, psd_names1, freqs = load_single_append(path, fnames1, typ='psd')
psd2, psd_stag2, psd_names2, freqs = load_single_append(path, fnames2, typ='psd')

psd_ns = get_unique_name_list(psd_names1, psd_names2)
mspe_ns = get_unique_name_list(mspe_names1, mspe_names2)
assert all([psd_ns[i] == mspe_ns[i] for i in range(len(psd_ns)) ])
assert all([all(mspe_stag1[i] == psd_stag1[i]) for i in range(len(psd_stag1))])
assert all([all(mspe_stag2[i] == psd_stag2[i]) for i in range(len(psd_stag2))])
del (psd_stag1, psd_stag2)

total_count_stag1 = [ len(mspe_stag1[i]) for i in range(len(mspe_stag1)) ]
total_count_stag2 = [ len(mspe_stag2[i]) for i in range(len(mspe_stag2)) ]

#mspe1, psd1, stag1, count_artifs1 = remove_20hz_artif(mspe1, psd1, mspe_stag1, mspe_names1, freqs, bad_sbjs_1)
#mspe2, psd2, stag2, count_artifs2 = remove_20hz_artif(mspe2, psd2, mspe_stag2, mspe_names2, freqs, bad_sbjs_2)
stag1 = mspe_stag1 #when no artif corr
stag2 = mspe_stag2 #when no artif corr

mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)
psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

#Get percentage of epochs different than [NREM, REM, WAKE] (run on data without 20Hz correct)
ratio_stag1 = 100 - np.asarray([ len(stag1[i]) for i in range(len(stag1)) ]).sum() / \
             float(np.asarray(total_count_stag1).sum() ) * 100
ratio_stag2 = 100 - np.asarray([ len(stag2[i]) for i in range(len(stag2)) ]).sum() / \
             float(np.asarray(total_count_stag2).sum() ) * 100



rel_psd1 = [psd1[i] / np.abs(np.sum(psd1[i], 0)) for i in range(len(psd1))]
rel_psd1 = [ np.log10(rel_psd1[i]) for i in range(len(psd1)) ]
rel_psd2 = [psd2[i] / np.abs(np.sum(psd2[i], 0)) for i in range(len(psd2))]
rel_psd2 = [ np.log10(rel_psd2[i]) for i in range(len(psd2)) ]
psd1 = rel_psd1
psd2 = rel_psd2
psd1 = [ psd1[i].reshape(-1, psd1[i].shape[-1]) for i in range(len(psd1)) ] # reshape
psd2 = [ psd2[i].reshape(-1, psd2[i].shape[-1]) for i in range(len(psd2)) ] #reshape

mspe1_ = [ mspe1[i ][:s, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

mspe2_ = [ mspe2[i ][:s, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

if setup == 'mspet1m3':
    if time == 2:
        data_pe, data_stag = mspe1, stag1
    elif time == 5:
        data_pe, data_stag = mspe2, stag2
    elif time == 'cat':
        data_pe = mspe1 + mspe2
        data_stag = stag1 + stag2
elif setup == 'psd':
    if time == 2:
        data_pe, data_stag = psd1, stag1
    elif time == 5:
        data_pe, data_stag = psd2, stag2
    elif time == 'cat':
        data_pe = psd1 + psd2
        data_stag = stag1 + stag2

# Index for plot sbj
idx_plot = 1
#assert names1[idx_plot].split('_')[0] == names2[idx_plot].split('_')[0]

# Get all metrics
perf  = classify_shuffle(data_pe, data_stag, idx_plot, myshow=True, check_mspe=True, null=False,
                            n_folds=n_folds, search=False)
#print accuracy
print np.asarray([perf[i][0] for i in range(len(perf))]).mean()

#print importances
#imp = dict()
#for s in [1,2,3]:
#    imp[s] = np.asarray([perf[i][-1][s] for i in range(len(perf))]).mean(0)


#write_pickle(perf, 'psd_cat_searched_scores.txt')

#store_perfs.extend([perf])

#write_pickle(perf, 'mspe5_acc_prec_rec2.txt')

#Aggregate folds
#accav = np.asarray([perf[i][0] for i in range(n_folds)]).mean(0)
#accav = np.asarray([perf[i][0] for i in range(len(perf))]).mean(0)
#print accav
#cmav = np.asarray([perf[i][1] for i in range(len(perf))]).mean(0)
#recall =  np.asarray([perf[i][2] for i in range(len(perf))]).mean(0)
#precission =  np.asarray([perf[i][3] for i in range(len(perf))]).mean(0)
#f1perclass =  np.asarray([perf[i][4] for i in range(len(perf))]).mean(0)
#cm_title = 'Confusion matrix'
#plot_confusion_matrix(cmav, ['NREM', 'REM', 'WAKE'], title=cm_title, normalize=True)
#store_perfs[setup].extend([perf])



#Run shuffling
nulliter  = 100
null_perfs = []

for idx in range(nulliter):
    perf_n = classify_shuffle(data_pe, data_stag, idx_plot, myshow=False, check_mspe=True, \
                                null=True, n_folds=n_folds, search=False)
    print(idx)
    null_perfs.append(perf_n)

write_pickle(null_perfs, 'null_week5.txt')

d = read_pickle('psd_cat_null_week5.txt')
dd = np.asarray([np.asarray([d[i][j][2] for j in range(len(d[0]))]) for i in range(len(d))])
