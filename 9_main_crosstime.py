'''
Run CROSS classification using custom function classify_shuffle_crosstime. Config params:
- mspe or psd data
- mspe scales (e.g 1-5)
- week2 or week5 session
- shuffling based null distribution or 'true' classification
- random search for hyperparameters or default classifier
'''
import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, load_single_append,write_pickle, read_pickle, plot_confusion_matrix,\
                        remove_20hz_artif, align_t1_t2_data)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pandas as pd
import pickle
import seaborn as sns
from classify_with_shuffling_crosstime import classify_shuffle_crosstime
from collections import defaultdict
from IPython.core.debugger import set_trace
from config import bad_sbjs_1, bad_sbjs_2

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]
n_folds = 2
five2two = False #if True cross generalization (5weeks train, 2weeks test) otherwise the opposite
store_perfs = []
s = 5 #taus

mspe1, mspe_stag1, mspe_names1, _ = load_single_append(path, fnames1, typ='mspet1m3')
mspe2, mspe_stag2, mspe_names2, _ = load_single_append(path, fnames2, typ='mspet1m3')
psd1, psd_stag1, psd_names1, freqs = load_single_append(path, fnames1, typ='psd')
psd2, psd_stag2, psd_names2, freqs = load_single_append(path, fnames2, typ='psd')

assert all([all(mspe_stag1[i] == psd_stag1[i]) for i in range(len(psd_stag1))])
assert all([all(mspe_stag2[i] == psd_stag2[i]) for i in range(len(psd_stag2))])
del (psd_stag1, psd_stag2)

mspe1, psd1, stag1, _ = remove_20hz_artif(mspe1, psd1, mspe_stag1, mspe_names1, freqs, bad_sbjs_1)
mspe2, psd2, stag2, _ = remove_20hz_artif(mspe2, psd2, mspe_stag2, mspe_names2, freqs, bad_sbjs_2)

mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)
psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

mspe1, mspe2, stag1, stag2 = align_t1_t2_data(mspe1, mspe2, stag1, stag2) #get matching subject only

psd1 = [ psd1[i].reshape(-1, psd1[i].shape[-1]) for i in range(len(psd1)) ] # reshape
psd2 = [ psd2[i].reshape(-1, psd2[i].shape[-1]) for i in range(len(psd2)) ] #reshape

mspe1_ = [ mspe1[i ][:s, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

mspe2_ = [ mspe2[i ][:s, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

# Index for a sbj to plot
idx_plot = 1

# Get all metrics
perf = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                    check_mspe=True, null=False, n_folds=n_folds, five2two=five2two, search=True)

# Save scores
write_pickle(perf, 'mspe52_cat_searched_scores.txt')


# Run classification on shuffled data (chance level)
nulliter  = 100
null_perfs = []

for idx in range(nulliter):
    perf_n = classify_shuffle_crosstime(mspe1, mspe2, stag1, stag2, myshow=False, \
                        check_mspe=True, null=True, n_folds=n_folds, five2two=five2two, search=False)
    print(idx)
    null_perfs.append(perf_n)
# Save chance scores
write_pickle(null_perfs, 'null_week25.txt')
