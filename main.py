import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        vis_PCA, classify_lso, classify_generalize_time,
                        merge_stages, count_stag, group_h0, vis_clf_probas,
                        align_t1_t2_data, write_pickle, read_pickle)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
from classify_with_shuffling_v2 import classify_shuffle, classify_shuffle_crosstime


sel_idxs = [1,2,3]
path = 'E:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

setup= 'mspet1m3'

'''
# SIMPLE GENERALIZATION BASED ON PE
pe, stag, _, _ = load_single_append(path, fnames1, typ=setup)
pe, stag = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)
#mypsd, stag, _ =  load_single_append(path, fnames2, typ = 'psd')
pred_pe, scores_pe, _, perf, names, _ = classify_lso(pe, stag, myshow=False, setup=setup, \
                                                            oversample=False)
#report.save(base_path+setup +'_current_report.html', overwrite=True)
print np.asarray(perf).mean(0)
#vis_clf_probas(pe, stag)

h0_gr, p, perf  = group_h0(pe, stag,  perf, times=10e4) #significance testing
'''

# SIMPLE GENERALIZATION BASED ON MSPE
mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

#mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
#mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

#Get classification perf. for given data set 2weeks vs 5 weeks)
data_pe, data_stag = mspe2, stag2
plt.ioff()
perf, report = classify_shuffle(data_pe, data_stag, myshow=False, check_mspe=True, null=False)
av_perf = perf.mean(0) # av. folds

#permutations to estimate signle subject chance
nulliter  = 100
nullperf = np.empty([len(av_perf), nulliter])
for i in range(nulliter):
    shuffled = classify_shuffle(data_pe, data_stag, myshow=False, check_mspe=True, \
                                null=True)
    nullperf[:,i] = shuffled.mean(0)  # av. folds
    print(i)

res = {'perf' : av_perf, 'null_distr' : nullperf}
'''bootstrapping skiped, simple averaging across ss. perfs. instead
#bootstrapping for null on a group level
nullgroupiter = 1000
nullperf_gr = np.zeros([nullgroupiter, 1])

for j in range(nullgroupiter):
    # sample with repl. subject performance and average
    nullperf_gr[j, :] = np.mean(nullperf[:, np.random.choice(range(nulliter))])

    r = nullperf_gr > np.asarray(perf).mean()
    #p = r.sum() / float(len(h0_gr))
    p = r.sum() / ( float(nullgroupiter) + 1 )
res = [perf, nullperf_gr, p]
'''
#write_pickle(res, 'perf_nulldistr2.txt')
#dd1 = read_pickle('perf_nulldistr1.txt')
#dd2 = read_pickle('sign_t2_mspe.txt')

################################################################################
# TIME GENERALIZATION BASED ON MSPE
'''
mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

mspe1, mspe2, stag1, stag2 = align_t1_t2_data(mspe1, mspe2, stag1, stag2) #get matches only

mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape
early2late = True
perf = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                    check_mspe=True, null=False, early2late=early2late)
print np.asarray(perf).mean()
##
#permutation for signle subject
nulliter  = 100
nullperf = np.empty([len(perf), nulliter])
for i in range(nulliter):
    nullperf[:,i] = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                                    check_mspe=True, null=True, early2late=early2late)
    print i

#bootstrapping for null on a group level
nullgroupiter = 1000
nullperf_gr = np.zeros([nullgroupiter, 1])
for j in range(nullgroupiter):
    # sample with repl. subject performance and average
    nullperf_gr[j, :] = np.mean(nullperf[:, np.random.choice(range(nulliter))])

    r = nullperf_gr > np.asarray(perf).mean()
    #p = r.sum() / float(len(h0_gr))
    p = r.sum() / ( float(nullgroupiter) + 1 )
res = [perf, nullperf_gr, p]
write_pickle(res, 'sign_t1t2_mspe_weightedF1.txt')
'''
