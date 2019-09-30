'''
Load and plot classification scores for PSD and MSPE (confusion matrix and box plots)
'''
import os
import numpy as np
from config import results_path
from matplotlib import pyplot as plt
from functional import read_pickle, plot_confusion_matrix
from IPython.core.debugger import set_trace
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu as mann
import matplotlib.gridspec as gridspec
from numpy import mean
from matplotlib.lines import Line2D

def my_load(fname):
    path = results_path + 'scores' + '\\acc_f1indiv\\' + fname
    scores = read_pickle(path)
    return scores

fname1 = 'mspe_cat_searched_scores.txt'
fname2 = 'psd_cat_searched_scores.txt'

mspe = my_load(fname1)
psd = my_load(fname2)

psdacc = [psd[i][0] for i in range(len(psd))]
mspeacc = [mspe[i][0] for i in range(len(mspe))]
u, p = mann(psdacc, mspeacc)

fig, ax =plt.subplots(1, figsize=(6,6))
df = pd.DataFrame({'psd' : psdacc, 'mspe':mspeacc})
df = pd.melt(df)
sns.boxplot(x='variable', y='value', data=df, order=['psd', 'mspe'], ax=ax)
sns.swarmplot(x='variable', y='value', data=df, order=['psd', 'mspe'], color='black',ax=ax)
ax.set(ylabel='Accuracy [%]', xlabel='', xticklabels=['PSD', 'MSPE'], ylim=[0.45, 0.75])
plt.tight_layout()
plt.savefig('psd_mspe_scores_acc.tif', dpi=300)


psdcm = np.asarray([psd[i][1] for i in range(len(psd))]).mean(0)
mspecm = np.asarray([mspe[i][1] for i in range(len(mspe))]).mean(0)
plot_confusion_matrix(psdcm, ['NREM', 'REM', 'WAKE'], title=None, normalize=True)
plt.savefig('psd_confusion.tif', dpi=300)
plot_confusion_matrix(mspecm, ['NREM', 'REM', 'WAKE'], title=None, normalize=True)
plt.savefig('mspe_confusion.tif', dpi=300)
