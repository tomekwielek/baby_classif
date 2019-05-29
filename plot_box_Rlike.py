import os
import numpy as np
from config import results_path
from matplotlib import pyplot as plt
from IPython.core.debugger import set_trace
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from numpy import mean
from matplotlib.patches import Patch


#df = pd.read_csv('H:\\BABY\\results\\old\\stat\\df_l_eps.csv')
df = pd.read_csv('H:\\BABY\\results\\mspe_allsbjs_alleeg_10epochs.csv')
#df = pd.read_csv('mspe.csv')

#drop 213 physio flat, see subjects_used.xlsx
#df = df[df['name_id_short'] != 213] #for PE analysis only
#df = df.dropna()

df = df[df['variable'] == 4]


fig, ax = plt.subplots()
box = sns.boxplot(x='stag', y='value', hue='time',data=df, linewidth=3, \
                ax=ax, whis=[5, 95], showfliers=False, dodge=True)
def modif_box(axes, facecolor, edgecolor, idxs):
    for i in idxs:
        mybox = box.artists[i]
        mybox.set_facecolor(facecolor)
        mybox.set_edgecolor(edgecolor)
        # Each box has several associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*5,i*5+5):
            ax = plt.gca()
            line = ax.lines[j]
            line.set_color(edgecolor)
            line.set_mfc(edgecolor)
            line.set_mec(edgecolor)
    return

modif_box(box, 'white', 'black', [0,2,4])
modif_box(box, 'white', 'red', [1,3,5])

sns.swarmplot(x='stag', y='value', hue='time',data=df, split=True, color='black', \
            size=4, alpha=0.7, ax=ax)

legend_elements = [Patch(facecolor='white', edgecolor='black',
                         label='week 2', linewidth=3),
                    Patch(facecolor='white', edgecolor='red',
                        label='week 5', linewidth=3)]
ax.legend(handles=legend_elements, loc='lower right')
#ax.set(ylabel= 'MSPE(scale=5)', xlabel='', xticklabels= ['NREM', 'REM', 'WAKE'], ylim=[1.2, 1.68])
ax.set(ylabel= 'MSPE(scale=4)', xlabel='', xticklabels= ['NREM', 'REM', 'WAKE'], ylim=[1.38, 1.71])

#ax.plot([-0.2, -0.2, 0.2, 0.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([-0.2, -0.2, 0.2, 0.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.plot([0.8, 0.8, 1.2, 1.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([0.8, 0.8, 1.2, 1.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.plot([1.8, 1.8, 2.2, 2.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([1.8, 1.8, 2.2, 2.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.text(-0.07, 1.645, 'p<.05', color='black', size=10)
#ax.text(0.93, 1.645, 'p<.05', color='black', size=10)
#ax.text(1.95, 1.645, 'ns', color='black', size=10)
ax.text(-0.07, 1.685, 'p<.05', color='black', size=10)
ax.text(0.93, 1.685, 'p=.06', color='black', size=10)
ax.text(1.95, 1.685, 'ns', color='black', size=10)
plt.rcParams.update({'font.size': 15})
#pyplot.locator_params(axis='y', nbins=8)
plt.show()



#get summary, descriptive
def get_stats(group):

    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean(), 'SD':group.std()}
df_dsc = df.groupby(['time', 'stag'])['value'].apply(get_stats).unstack()
df_dsc = df_dsc[['count', 'mean', 'SD', 'min', 'max']]
df_dsc = df_dsc.round(3)
