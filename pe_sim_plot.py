import numpy as np
import matplotlib.pyplot as plt
from pyentrp import entropy as ent
from IPython.core.debugger import set_trace

# from https://www.elenacuoco.com/2016/07/31/simulating-time-series/
#ARMA
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from pyentrp import entropy as ent
np.random.seed(12345)

sfreq = 4
window = 15
segments = 1
nsample = sfreq * window * segments
x1 = np.linspace(0, 30, nsample)

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]

raw = arma_generate_sample(arparams, maparams, nsample)

def plot_raw(data):
    fig, ax = plt.subplots(figsize=(7,4))
    times = range(len(data))
    ax.plot(times, data, linestyle='-', marker='o', color='black')
    ax.axvline(times[0], color='b', alpha=0.5, linestyle='--')
    ax.axvline(times[2], color='b', alpha=0.5, linestyle='--')
    ax.axvline(times[4], color='b', alpha=0.5, linestyle='--')
    ax.axvline(times[6], color='b', alpha=0.5, linestyle='--')
    ax.set(xticks=[], yticks=[])
    plt.show()

def get_granulated(data, scale):
    gr = ent.util_granulate_time_series(raw, scale)
    return gr


raw1 = get_granulated(raw, scale=2)

plot_raw(raw)
plot_raw(raw1)







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
