import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from functional import read_pickle
from IPython.core.debugger import set_trace
from scipy.stats import mannwhitneyu as mann
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

path = 'H:\\BABY\\results\\scores\\'
actual_fname = 'mspe_vs_psd_cat.txt'
null_fname_mspe = 'mspe_cat_null.txt'

perf = read_pickle(path + actual_fname)
null_perf_mspe = read_pickle(path + null_fname_mspe)


acc_psd = np.asarray([perf['psd'][0][i][0] for i in range(len(perf['psd'][0]))]) * 100
acc_mspe = np.asarray([perf['mspet1m3'][0][i][0] for i in range(len(perf['mspet1m3'][0]))]) * 100
f1_psd = np.asarray([perf['psd'][0][i][4] for i in range(len(perf['psd'][0]))]) * 100
f1_mspe = np.asarray([perf['mspet1m3'][0][i][4] for i in range(len(perf['mspet1m3'][0]))]) * 100

null_acc_mspe =np.asarray([ [np.asarray([null_perf_mspe[j][i][0] for i in range(20)]) * 100] for j in range(100)])


def get_pands(d, time):
    df = pd.DataFrame(d)
    if d.ndim == 2:
        df = df.rename(columns={0:'NREM', 1:'REM', 2:'WAKE'})
    elif d.ndim ==1:
        df = df.rename(columns={0:'acc'})
    df['time'] = [time] * len(d)
    return df


acc_psd = get_pands(acc_psd, time='psd')
acc_mspe = get_pands(acc_mspe, time='mspe')
f1_psd = get_pands(f1_psd, time='psd')
f1_mspe = get_pands(f1_mspe, time='mspe')

U, pval = mann(acc_mspe['acc'], acc_psd['acc'])

f1_cat = pd.concat([f1_psd, f1_mspe])
f1_cat_melt = pd.melt(f1_cat, id_vars= 'time')


gs1 = gridspec.GridSpec(1, 8)
gs1.update(left=0.1, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[:, :3])
ax2 = plt.subplot(gs1[:, 3:4])
ax3= plt.subplot(gs1[:, 4:7])
ax4= plt.subplot(gs1[:, 7:8])
gs2 = gridspec.GridSpec(1, 8)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax5 = plt.subplot(gs2[:, :8])
sns.set_style('whitegrid')

palette ={'psd':"C0",'mspe':"C1"}


sns.barplot(x='time', y = 'acc', data=acc_psd,  ax=ax1, estimator=mean, ci=95, color='grey', \
            edgecolor='black')
#ax1.axhline(y=mean(null_acc_52.ravel()), linestyle='--', color='black', alpha=0.5)
#ax2.hist(null_acc_52.ravel(), bins=20, orientation='horizontal', color='darkgrey', edgecolor='black')

sns.barplot(x='time', y = 'acc', data=acc_mspe,  ax=ax3, estimator=mean, ci=95, color='grey', \
            edgecolor='black')
ax3.axhline(y=mean(null_acc_mspe.ravel()), linestyle='--', color='black', alpha=0.5)
ax4.hist(null_acc_mspe.ravel(), orientation='horizontal', bins=20, color='darkgrey', edgecolor='black')

sns.barplot(x='time', y = 'value', hue='variable', data=f1_cat_melt,  ax=ax5, estimator=mean,
            ci=95, edgecolor='black')

ax1.set(ylim=[0,90], xlabel='',  ylabel='Accuracy [%]', xticks=[0.3], xticklabels=['week2'])
ax2.set(ylim=[0,90],  xlabel='', yticklabels=[],  xticks=[])
ax3.set(ylim=[0,90], yticklabels=[], xlabel='', ylabel='', xticks=[0.3], xticklabels=['week5'])
ax4.set(ylim=[0,90],  xlabel='', yticklabels=[], xticks=[])
ax5.set(ylim=[0,90], ylabel='F1 [%]', xlabel='', yticklabels=[], xticks=[0,1], \
        xticklabels=['PSD', 'MSPE'])
ax5.legend(prop=dict(size=12))
legend_elements = [Line2D([0], [0], color='grey', lw=2, label='Chance', linestyle='--')]
ax1.legend(handles=legend_elements, loc='best', prop={'size': 8})
