import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from functional import read_pickle
from IPython.core.debugger import set_trace
from scipy.stats import mannwhitneyu as mann
import seaborn as sns

path = 'H:\\BABY\\results\\20.10_res\\'
path = path +  'mspe_vs_psd_cat.txt'
perf = read_pickle(path)
n_folds = 20

acc = {'mspet1m3' : [], 'psd' : []}
for k in perf.keys():
    for i in range(20):
        #mean folds
        acc[k].extend([np.asarray([perf[k][i][j][0] for j in range(len(perf[k][0]))]).mean(0)])


U, pval = mann(acc['mspet1m3'], acc['psd'])


df = pd.DataFrame(acc)
df = df.melt()
fig, ax = plt.subplots(1,1, figsize=(6, 6))
#sns.barplot(x='variable', y = 'value', data=df, ci='sd', ax=ax)
sns.boxplot(x='variable', y = 'value', data=df,  ax=ax, width=0.4)
sns.swarmplot(x='variable', y = 'value', data=df,  ax=ax, color ='black')
ax.set( ylabel='Accuracy [%]', xlabel='', xticks=[0,1], yticks=[0.64, 0.65, 0.66, 0.67, 0.68], \
        xticklabels=['MSPE', 'PSD'])


matplotlib.rcParams.update({'font.size': 15})
