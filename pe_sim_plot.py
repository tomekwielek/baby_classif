import numpy as np
import matplotlib.pyplot as plt
from pyentrp import entropy as ent
from IPython.core.debugger import set_trace

# from https://www.elenacuoco.com/2016/07/31/simulating-time-series/
'''
#ARMA
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)

sfreq = 128
window = 30
segments = 2
nsample = sfreq * window * segments
x1 = np.linspace(0, 30, nsample)

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]

raw = arma_generate_sample(arparams, maparams, nsample)
#fig, ax = plt.subplots()
#ax.plot(x1, raw)



def compute_pe_segm(raw, embed=3, tau=1, window=window, mspe=True):
    length_dp = len(raw)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    #set_trace()
    chan_len = 1
    store_segm = []
    data = raw
    #data = raw[np.newaxis, :]

    scale = 4
    m = np.zeros((scale, no_epochs)) # multiscale entropy

    course_store = []
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        print e_.shape
        data = np.delete(data, np.s_[:window_dp])

        m[:,i], course = ent.multiscale_permutation_entropy(e_, m=3, delay=1, scale=scale)
        course_store.append(course)
        del e_
    return m, course_store
pe, course  = compute_pe_segm(raw, embed=3, tau=1, window=window, mspe=True)

'''

from read_raw import read_raw, preproces, load_raw_and_stag, compute_pe_segm
from functional import map_stag_with_raw
from config import raw_path, stag_fname
import pandas as pd
import os

fnames_ = sorted(np.array(filter(lambda x: x.endswith('.edf'), os.listdir(raw_path))))
fnames = [i.split('.edf')[0] for i in fnames_]

s = pd.read_excel(stag_fname, sheetname='St_Pr_corrected_35min')
s = my_rename_col(s)
s = s.drop(s.columns[[0, 1, 2]], axis=1) #drop cols like 'Condition', 'Minuten', etc

idfs = map_stag_with_raw(fnames, s=s, sufx='ref100')
idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls
