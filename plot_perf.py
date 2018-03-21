import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
path = 'H:\\BABY\\results\\perf_raw\\'
perf1 = pickle.load(open(path+'scores_t1_t1m3_fw_kfold2.txt', 'rb'))
perf2 = pickle.load(open(path+'scores_t2_t1m3_fw_kfold2.txt', 'rb'))
f1s_each = pickle.load(open(path+'f1s_each_t2.txt', 'rb'))

def melt_perfs(perf1, perf2):
    df1 = pd.DataFrame(dict([(key, perf1[key]) for key in ['pval', 'scores']]))
    df1['time'] = [1] * len(df1)
    df2 = pd.DataFrame(dict([(key, perf2[key]) for key in ['pval', 'scores']]))
    df2['time'] = [2] * len(df2)
    df = pd.concat([df1, df2], ignore_index=False,axis=0 )
    return df

df = melt_perfs(perf1, perf2)
h0_1 = perf1['h0']
h0_2 = perf2['h0']

fig, axes = plt.subplots(2,1, sharex= True)
sc1 = np.asarray(perf1['scores']).mean()
sns.distplot(h0_1, ax = axes[0])
axes[0].axvline(x = sc1, c='red', linewidth='4')
axes[0].set_ylabel('Density')
axes[0].text(sc1-.055, 6.7, 'p=ns.' %perf1['pval'], bbox=dict(facecolor='red', alpha=0.5))
axes[0].set_ylim([0,10])
axes[0].set_title('time1'.upper())

sc2 = np.asarray(perf2['scores']).mean()
sns.distplot(h0_2, ax = axes[1])
axes[1].axvline(x = sc2, c='red', linewidth='4')
axes[1].set_ylabel('Density')
axes[1].set_xlabel('Classification performance [F1-score]')
axes[1].text(sc2+.01, 6.7, 'p=%.3f' %perf2['pval'], bbox=dict(facecolor='red', alpha=0.5))
axes[1].set_ylim([0,10])
axes[1].set_title('time2'.upper())

'''
Plot f1 score for each sleep stage (averaged across subjects)
'''
def plot_f1s_each(d):
    df =pd.DataFrame(f1s_each)
    df = df.melt()
    fig, ax = plt.subplots()
    sns.barplot(x='variable', y='value', estimator=mean, data=df, ci='sd', ax=ax)
    ax.set_xticklabels(['NREM', 'REM', 'WAKE'])
    ax.set_xlabel('stage')
    ax.set_ylabel('Classification performance [F1-score]')
