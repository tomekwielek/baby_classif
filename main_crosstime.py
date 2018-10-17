import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, load_single_append,
                        merge_stages, count_stag, align_t1_t2_data, write_pickle, read_pickle, plot_confusion_matrix)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
from classify_with_shuffling_v2_crosstime import classify_shuffle_crosstime
from collections import defaultdict
from IPython.core.debugger import set_trace

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]
n_folds = 2
five2two = False #if True cross gen: 5weeks - 2weeks otherwise the oposite

setup= 'mspet1m3'

if setup == 'mspet1m3':
    # CROSS TIME GENERALIZATION BASED ON MSPE
    mspe1, stag1, names1, _ = load_single_append(path, fnames1, typ=setup)
    mspe2, stag2, names2, _ = load_single_append(path, fnames2, typ=setup)
    mspe1, stag1 = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
    mspe2, stag2 = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)

    mspe1, mspe2, stag1, stag2 = align_t1_t2_data(mspe1, mspe2, stag1, stag2) #get matching subject only

    mspe1_ = [ mspe1[i ][:4, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
    mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

    mspe2_ = [ mspe2[i ][:4, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
    mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

    # Get actual scores
    f1, f1_indiv = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                        check_mspe=True, null=False, n_folds=n_folds, five2two=five2two, search=True)

    #Run shuffling
    nulliter  = 1000
    null_f1 = np.empty([nulliter])
    null_f1_indiv = np.empty([nulliter, 3])
    for idx in range(nulliter):
        f1_, f1_indiv_ = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                            check_mspe=True, null=True, n_folds=n_folds, five2two=five2two, search=False)
        null_f1[idx] = f1_
        null_f1_indiv[idx] = f1_indiv_
        print(idx)


    #Save
    write_pickle((null_f1, f1), 'two2five_f1.txt')
    write_pickle((null_f1_indiv, f1_indiv), 'two2five_f1_indiv.txt')

    #Test saving
    #n, a = read_pickle('two2five_recall.txt')
