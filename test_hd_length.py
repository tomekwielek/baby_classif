import os
import numpy as np
from config import myload, base_path, paths
from mne import io
import pickle
from IPython.core.debugger import set_trace
import mne
from config import markers35, markers27, stag_fname, length_issue_sbjs
import pandas as pd
from functional import (my_rename_col, map_stag_with_raw, load_single_append,
                            select_class_to_classif)

sheet = 'St_Pr_corrected_27min'

if '35' in sheet:
    markers = markers35
elif '27' in sheet:
    markers = markers27

mff_path = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\'
edf_path = 'H:\\BABY\\data\\'

fnames_edf_ = sorted(list(filter(lambda x: x.endswith('ref100.edf'), os.listdir(edf_path))))
fnames_edf = [i.split('.edf')[0] for i in fnames_edf_]
fnames_mff = sorted(list(filter(lambda x: x.endswith('.mff'), os.listdir(mff_path))))

ss = pd.read_excel(stag_fname, sheet_name=sheet)
ss = my_rename_col(ss)
ss = ss.drop(ss.columns[[0, 1, 2]], axis=1) #drop cols like 'Condition', 'Minuten', etc

idfs = map_stag_with_raw(fnames_mff, ss, sufx='.mff')
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls

match_names = [] #store names where length matches
for k, v in sorted(idfs.items()):
    k_short = k.split('_S')[0]
    if not os.path.isfile(paths(typ='psd', sbj=k_short)):
        print('Folder is empty')
        continue
    stag, psd_hd, freqs = myload(typ='psd_hd', sbj=k_short)
    _, psd, _ = myload(typ='psd', sbj=k_short)
    print('HD count= {}, sbj {}'. format(psd_hd.shape[-1], k_short))
    print('OLD count= {}, sbj {}'. format(psd.shape[-1], k_short))
    if psd_hd.shape[-1] == psd.shape[-1]:
        match_names.append(k)


##############################################################################
