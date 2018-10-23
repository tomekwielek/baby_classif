import os
import numpy as np
from config import results_path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from functional import read_pickle
from IPython.core.debugger import set_trace
from matplotlib.patches import Patch

def my_load(fname):
    path = results_path + 'scores' + '\\acc\\' + fname
    scores = read_pickle(path)
    return scores

fnames_dict = {'five' : ({'acc_score' : 'mspe5_acc_prec_rec.txt'},
                            {'idx' : (1,1)}),

                 }

#PLOT AV S1 SCORES
def plot_f(null_f, f, title, ax):
    null_f = null_f * 100
    my_median = int(median(null_f))
    f = f *100
    diff = f - my_median
    ax.hist(null_f, 20, color='black', density =True)
    ax.axvline(f, color='red')
    ax.axvspan(my_median, f, alpha=diff/100., color='blue')
    ax.set_xlim([0,100])
    ax.set_xticks([my_median, int(f)], [])

    ax.tick_params(axis='x', which='major', labelsize='xx-large', labelcolor='black', rotation=0)
    ax.set_yticks([], [])
    del f, null_f
    return ax

fig, axes = plt.subplots(2,2, sharex=False, sharey=False)
for sn in fnames_dict.keys():
    null_f, f = my_load(fnames_dict[sn][0]['acc_score'])
    title = 'F score classification: {}'.format(sn)
    idx = fnames_dict[sn][1]['idx']
    print idx

    plot_f(null_f, f, title, axes[idx])

#fig.text(0.5, 0.01, 'F1 score [%]', ha='center')
legend_elements = [Line2D([0], [0], color='red', lw=3, label='Actual'), Line2D([0], [0], color='black', lw=3, label='Chance'),
Patch(facecolor='blue', edgecolor='b', label='Difference', alpha=0.6) ]

fig.legend(handles=legend_elements, loc='upper right', prop={'size': 14})
fig.show()



fig = plt.figure(figsize=(15, 4))
outer = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.8)

for i, sn in enumerate( fnames_dict.keys() ):
    sleep_classes = ['NREM', 'REM', 'WAKE']
    null_f, f = my_load(fnames_dict[sn][1]['f_indivs'])
    null_f = null_f * 100
    f = f *100
    idx = fnames_dict[sn][2]['idx']

    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[idx], wspace=0.1, hspace=0.1)

    for j, tit in enumerate(sleep_classes):
        ax = plt.Subplot(fig, inner[j])

        f1s = null_f[:,j]
        my_median = int(median(f1s))
        #my_max = int(max(f1s))
        my_f1 =  int(f[j])
        ax.hist(f1s,  color='black')
        ax.axvline(f[j], color='red')
        diff = my_f1 - my_median
        ax.axvspan(my_median, my_f1, alpha=diff/100., color='blue')
        ax.set_xlim([0,100])
        ax.set_xticks([my_median, my_f1], [])

        ax.tick_params(axis='x', which='major', labelsize='xx-large', labelcolor='black', rotation=45)
        ax.set_yticks([], [])
        if idx in [(0,0), (0,1)]:
            ax.set_title(tit, fontsize=17)

        fig.add_subplot(ax)
#fig.text(0.5, 0.001, 'F1 score [%]', ha='center', fontsize=15)
legend_elements = [Line2D([0], [0], color='red', lw=3, label='Actual'), Line2D([0], [0], color='black', lw=3, label='Chance'),
Patch(facecolor='blue', edgecolor='b', label='Difference', alpha=0.6) ]

fig.legend(handles=legend_elements, loc='upper right', prop={'size': 14})
fig.show()


'''
#compute pvalues for accuracy
r_acc = null_acc > acc
p_acc = r_acc.sum() / ( float(nulliter) + 1 )
#compute pvalues for recall
r_r = [null_recall[:,i] > recall[i] for i in range(3) ]
p_r = [ r_r[i].sum() / ( float(nulliter) + 1 ) for i in range(3) ]
#compute pvalues for precission
r_p = [null_precision[:,i] > precision[i] for i in range(3) ]
p_p = [ r_p[i].sum() / ( float(nulliter) + 1 ) for i in range(3) ]
print 'pv acc {}, \n pv recall {},\n pv precision {}\n'.format(p_acc, p_r, p_p)
'''
