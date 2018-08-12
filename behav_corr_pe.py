import os
import numpy as np
from config import myload, base_path, paths, raw_path
from functional import (select_class_to_classif, cat_pe_stag, load_single_append)
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
fnames = [f for f in fnames if f.endswith('2')]
setup= 'pet1m3'
store_scores = dict()
store_pval = dict()
pe, stag, _ = load_single_append(path, fnames, typ = setup)
pe, stag = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)

pe_av = [pe[i].mean(0) for i in range(len(pe))]
#average channels
for i in range(len(pe)):
    stag[i]['pe'] = pe_av[i]
#average for each stag and std of stages
store = []
for i in range(len(pe)):
    av_s = stag[i].groupby(['numeric']).mean()
    no_s = len(av_s)
    #if no_s == 3:
    d = dict()
    d['pe'] = av_s
    d['name'] = stag[i].columns[0].split('_')[0]
    store.append(d)

df = pd.DataFrame(std_s).rename(columns = {'pe' : 'std'})

'''
load behav data
'''
scores_path = 'H:\\BABY\\results\\perf_raw\\'
f_name = raw_path + '\\behav\\Bayley_MS_Tomek.xlsx'
scores = pd.read_csv(scores_path + 'cross_time_scores.csv')

bley = pd.read_excel(f_name, sheet_name='Ubersicht')
ci = pd.read_excel(f_name, sheetname='CI_Details', skiprows=2)
bley = bley.dropna()
ci = ci.dropna()
bley['VPN'] = bley['VPN'].copy().astype(int).astype(str)
ci['VPN'] = ci['VPN'].copy().astype(str)
scores['name'] = scores['name'].copy().astype(str)
bley_sub = bley[bley['VPN'].isin(scores['name'].tolist())]
ci_sub = ci[ci['VPN'].isin(scores['name'].tolist())].reset_index()

scores_sub_bley = scores[scores['name'].isin(bley_sub['VPN'].tolist())]
scores_sub_ci = scores[scores['name'].isin(ci_sub['VPN'].tolist())].reset_index()

#combine pe data with behav
df['bley'] = bley[bley['VPN'].isin(df['name'].tolist())]['Unnamed: 3']
df.dropna()
