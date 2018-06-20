import pandas as pd
from config import results_path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
path = results_path + 'stat\\'

df_long = pd.read_csv(path + 'df_long_psd.csv')
nrempv = pd.read_csv(path + 'nrempv.csv')
rempv = pd.read_csv(path + 'rempv.csv')
wakepv = pd.read_csv(path + 'wakepv.csv')

freqs = np.loadtxt('freqs.txt')
def plot_stages_time_hued(df):
    fig, axes = plt.subplots(2,3, sharey=False, figsize = (10,5))
    old_ticks = axes[0, 0].get_xticks()
    for axi, s in enumerate(['nrem', 'rem', 'wake']):
        sns.pointplot(x='variable', y = 'value', hue='time', ci=95, \
                        data=df[df['stag']==s], ax = axes[0,axi])
        #old_ticks = axes[0, axi].get_xticks()
        axes[0,axi].set(xticks=old_ticks[::10], xticklabels=np.rint(freqs)[::10])
        axes[0,axi].set_title(s.upper())
        axes[0,axi].set_xlabel('freqs')
        axes[0,axi].set_ylabel('Log Relative Power')
    for axi, s in enumerate([nrempv, rempv, wakepv]):
        axes[1, axi].plot(freqs[::3], s['corrected'], linestyle='dotted')
        #old_ticks2 = axes[1, axi].get_xticks()
        axes[1,axi].set(xticks=old_ticks[::10], xticklabels=np.rint(freqs)[::10])
        axes[1,axi].set_xlabel('freqs')
        axes[1,axi].set_ylabel('pval')
    plt.show()

plot_stages_time_hued(df_long)
