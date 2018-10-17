import os
import numpy as np
from config import results_path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from functional import read_pickle
from IPython.core.debugger import set_trace

def my_load(fname):
    path = results_path + 'scores' + '\\' + fname
    scores = read_pickle(path)
    return scores

fnames_dict = {'five' : ({'accuracy' : 'five_acc.txt'},
                            {'precission' : 'five_precission.txt'},
                            {'recall' : 'five_recall.txt'},
                            {'idx' : (1,1)}),
               'two': ({'accuracy' : 'two_acc.txt'},
                        {'precission' : 'two_precission.txt'},
                        {'recall' : 'two_recall.txt'},
                        {'idx' : (0,0)}),
               'five2two' : ({'accuracy' : 'five2two_acc.txt'},
                         {'precission' : 'five2two_precission.txt'},
                         {'recall' : 'five2two_recall.txt'},
                         {'idx' : (1,0)}),
                'two2five' : ({'accuracy' : 'two2five_acc.txt'},
                           {'precission' : 'two2five_precission.txt'},
                           {'recall' : 'two2five_recall.txt'},
                           {'idx' : (0,1)})
                 }

#PLOT ACCURACIES
def plot_acc(null_acc, acc, title, ax, legend=False):
    ax.hist(null_acc*100, 20, color='black', density =True)
    ax.axvline(acc*100, color='red')
    ax.set_xlim([0,100])
    #ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    if legend:
        legend_elements = [Line2D([0], [0], color='red', lw=3, label='Actual'),
                            Line2D([0], [0], color='black', lw=3, label='Shuffled')]
        ax.legend(handles=legend_elements, loc='upper right')
    return ax

fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
for sn in fnames_dict.keys():
    null_acc, acc = my_load(fnames_dict[sn][0]['accuracy'])
    title = 'Accuracy for classification: {}'.format(sn)
    idx = fnames_dict[sn][3]['idx']
    legend=True if idx==(0,1) else False
    plot_acc(null_acc, acc, title, axes[idx], legend=legend)

fig.text(0.5, 0.01, 'Accuracy score [%]', ha='center')


def my_f1score(p, r):
    assert p.shape == r.shape
    fscores = np.ones(p.shape)
    if p.shape[0] > 1 and p.shape[0]  != 3: #null
        for ss in range(3):
            fscores[:,ss] = np.asarray([ (2*p[i][ss]*r[i][ss]) / sum((p[i][ss], r[i][ss])) for i in range(len(p)) ])
    elif p.shape[0] == 3: #actual
        fscores = np.asarray([(2*p[i]*r[i]) / sum((p[i], r[i])) for i in range(len(p)) ] )
    return fscores


#plot recall and precission
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(15, 4))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

for i, sn in enumerate( fnames_dict.keys() ):
    null_precission, precission = my_load(fnames_dict[sn][1]['precission'])
    null_recall, recall = my_load(fnames_dict[sn][2]['recall'])

    fscores = my_f1score(precission, recall)
    null_fscores = my_f1score(null_precission, null_recall)

    idx = fnames_dict[sn][3]['idx']

    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[idx], wspace=0.1, hspace=0.1)

    for j in range(3):
        ax = plt.Subplot(fig, inner[j])

        f1s = null_fscores[:,j][~np.isnan(null_fscores[:,j])]
        ax.hist(f1s,  color='black')
        ax.axvline(fscores[j], color='red')
        ax.set_xlim([0,1])

        ax.yaxis.set_major_formatter(plt.NullFormatter())
        if j==2 and idx == (0,1):
            legend_elements = [Line2D([0], [0], color='red', lw=3, label='Actual'),
                                Line2D([0], [0], color='black', lw=3, label='Shuffled')]
            ax.legend(handles=legend_elements, loc='upper right')
        fig.add_subplot(ax)

fig.show()

def plot_f1(null_f1, f1, title, axes, legend=False):
    sleep_classes = ['NREM', 'REM', 'WAKE']
    for i, ss in enumerate(sleep_classes):
        ax = plt.Subplot(fig, axes[i])
        #axes[i].hist(null_f1[:,i]*100, 20, color='black', density =True)
        ax.hist(null_f1[:,i]*100, 20, color='black', density =True)
        #axes[i].axvline(f1[i]*100, color='red')
        #axes[i].set_xlim([0,100])
        #ax.set_title(title)
        #axes[i].yaxis.set_major_formatter(plt.NullFormatter())
        #if legend:
        #    legend_elements = [Line2D([0], [0], color='red', lw=3, label='Actual'),
        #                        Line2D([0], [0], color='black', lw=3, label='Shuffled')]
        #    ax.legend(handles=legend_elements, loc='upper right')
    #return axes
    return ax

fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
plot_f1(null_fscores, fscores, title, axes, legend=False)

for sn in fnames_dict.keys():
    null_precission, precission = my_load(fnames_dict[sn][1]['precission'])
    null_recall, recall = my_load(fnames_dict[sn][2]['recall'])

    fscores = my_f1score(precission, recall)
    null_fscores = my_f1score(null_precission, null_recall)






    fig, axes = plt.subplots(2,3, sharex=True, sharey=True, figsize=(10,5))
    for i, ss  in enumerate(['NREM', 'REM', 'WAKE']):
        axes[0, i].hist(null_recall[:,i]*100, 20 )
        axes[0, i].axvline(recall[i]*100)
        axes[0, i].set_title(ss)
        axes[0, 0].set_ylabel('Recall score')
        axes[1, i].hist(null_precission[:,i]*100, 20 )
        axes[1, i].axvline(precission[i]*100)
        axes[1, 0].set_ylabel('Precision score')
        axes[1,i].set_xlim([0,100])
    plt.show()

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
