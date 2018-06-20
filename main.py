import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        vis_PCA, classify_lso, classify_generalize_time,
                        merge_stages, count_stag, subject_h0, group_h0, vis_clf_probas,
                        align_t1_t2_data)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
sel_idxs = [1,2,3]
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

setups= ['mspet1m3']
#sheet = 'St_Pr_corrected_27min'
store_scores = dict()
store_pval = dict()
for setup in setups:
    #read_raw(setup, sheet=sheet)
    # CROSS TIME GENERALIZATION
    #pe1, stag1, _ = load_single_append(path, fnames1, typ = setup)
    #pe2, stag2, _ = load_single_append(path, fnames2, typ = setup)

    #pe1, stag1 = select_class_to_classif(pe1, stag1, sel_idxs=sel_idxs)
    #pe2, stag2 = select_class_to_classif(pe2, stag2, sel_idxs=sel_idxs)

    #pe1, pe2, stag1, stag2 = align_t1_t2_data(pe1, pe2, stag1, stag2) #filter matches only
    #report, pred1, pred2, f1_1, f1_2, _, n1, n2 = classify_generalize_time(pe1, stag1, pe2, stag2,\
    #                                                                show=True, setup=setup)
    '''
    # SIMPLE GENERALIZATION
    pe, stag, _, _ = load_single_append(path, fnames2, typ=setup)
    pe, stag = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)
    #mypsd, stag, _ =  load_single_append(path, fnames2, typ = 'psd')
    pred_pe, scores_pe, _, perf, names, _ = classify_lso(pe, stag, myshow=False, setup=setup, \
                                                                oversample=False)
    #report.save(base_path+setup +'_current_report.html', overwrite=True)
    print np.asarray(perf).mean(0)
    #vis_clf_probas(pe, stag)

    h0_gr, p  = group_h0(pe, stag,  perf, times=10e4) #significance testing
    '''

    # SIMPLE GENERALIZATION BASED ON MSPE
    mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
    mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
    mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
    mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

    mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
    mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

    mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
    mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

    #pred_mspe1, scores_mspe1, _, f1s1, names1, f1s_each1 = classify_lso(mspe1, stag1, myshow=False, setup=setup, \
    #                                                            oversample=False)
    pred_mspe2, scores_mspe2, report, f1s2, names2, f1s_each2 = classify_lso(mspe2, stag2, myshow=True, setup=setup, \
                                                                oversample=False, mysave=True)
    #report.save(base_path+setup +'_current_report.html', overwrite=True)
    print np.asarray(f1s2).mean()

    h0_gr, p  = group_h0(mspe1, stag1,  f1s1, times=10e4)
