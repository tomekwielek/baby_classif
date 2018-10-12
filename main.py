import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        count_stag, vis_clf_probas, write_pickle, read_pickle, plot_confusion_matrix)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
from classify_with_shuffling_v2 import classify_shuffle
from collections import defaultdict
from IPython.core.debugger import set_trace

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]
time = 'cat' # if 2 week2, elif 5 week5, elif 'cat' concatenate
n_folds = 2

#setup= 'mspet1m3'
#setup = 'psd'
#iterate to compare statistically psd vs mspe based
store_acc = {'mspet1m3' : [], 'psd':[]}

for setup in ['mspet1m3', 'psd']:
    acc_iter = []
    for i in range(20):
        if setup == 'mspet1m3':
            # SIMPLE GENERALIZATION BASED ON MSPE
            mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
            mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
            mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
            mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

            mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
            mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

            mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
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
            if time == 2:
                data_pe, data_stag = psd1, stag1
            elif time == 5:
                data_pe, data_stag = psd2, stag2
            elif time == 'cat':
                data_pe = psd1 + psd2
                data_stag = stag1 + stag2

        plt.ioff()

        perf, report = classify_shuffle(data_pe, data_stag, myshow=False, check_mspe=True, null=False,
                                    n_folds=n_folds, search=True)
        #agregate over folds
        acc = np.asarray([perf[i][0] for i in range(n_folds)]).mean(0)
        store_acc[setup].extend([acc])

write_pickle(store_acc, 'psd_vs_mspe_scores_cat.txt')

#compare psd vs mspe
from scipy.stats import mannwhitneyu as mann
U, pval = mann(store_acc['mspet1m3'], store_acc['psd'])




cmx = np.asarray([np.vstack(perf[i][1]) for i in range(n_folds)]).sum(0)
recall = np.asarray([np.vstack(perf[i][2]) for i in range(n_folds)]).mean(0)
precision = np.asarray([np.vstack(perf[i][3]) for i in range(n_folds)]).mean(0)
#plot cm
cm_title = 'Confusion matrix' #, F1={:4.2f}'.format(acc)
plot_confusion_matrix(cmx, ['NREM', 'REM', 'WAKE'], title=cm_title, normalize=True)

#Run shuffling
nulliter  = 1000
null_acc = np.empty([nulliter])
null_recall = np.empty([nulliter,3])
null_precision = np.empty([nulliter,3])

for idx in range(nulliter):
    perf, _ = classify_shuffle(data_pe, data_stag, myshow=False, check_mspe=True, \
                                null=True, n_folds=n_folds, search=False)
    null_acc[idx] = np.asarray([perf[i][0] for i in range(n_folds)]).mean(0)
    null_recall[idx] = np.asarray([perf[i][2] for i in range(n_folds)]).mean(0)
    null_precision[idx] = np.asarray([perf[i][3] for i in range(n_folds)]).mean(0)
    print(idx)

#plot accuracy
plt.hist(null_acc*100, 20, color='black', density =True)
plt.axvline(acc*100, color='black')
plt.ylabel('Accuracy score')
plt.xlim([0,100])
plt.show()

#plot recall and precission
fig, axes = plt.subplots(2,3, sharex=True, sharey=True, figsize=(10,5))
for i, ss  in enumerate(['NREM', 'REM', 'WAKE']):
    axes[0, i].hist(null_recall[:,i]*100, 20 )
    axes[0, i].axvline(recall[i]*100)
    axes[0, i].set_title(ss)
    axes[0, 0].set_ylabel('Recall score')
    axes[1, i].hist(null_precision[:,i]*100, 20 )
    axes[1, i].axvline(precision[i]*100)
    axes[1, 0].set_ylabel('Precision score')
    axes[1,i].set_xlim([0,100])
plt.show()

#compute pvalues for accuracy
r_acc = null_acc > acc
p_acc = r_acc.sum() / ( float(nulliter) + 1 )
#compute pvalues for recall
r_r = [null_recall[:,i] > recall[i] for i in range(3) ]
p_r = [ r_r[i].sum() / ( float(nulliter) + 1 ) for i in range(3) ]
#compute pvalues for precission
r_p = [null_precision[:,i] > precision[i] for i in range(3) ]
p_p = [ r_p[i].sum() / ( float(nulliter) + 1 ) for i in range(3) ]
print 'pv acc {}, \n pv recall {},\n pv precision {}\n'.format(p_acc, p_r, p_p)



'''
#write_pickle(res, 'perf_nulldistr2.txt')
#dd1 = read_pickle('perf_nulldistr1.txt')
#dd2 = read_pickle('sign_t2_mspe.txt')
