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
'''
fnames_dict = {'five' : ({'perf' : 'mspe5_acc_prec_rec.txt'},
                            {'idx' : (1,1)}),
            'two' : ({'perf' : 'mspe2_acc_prec_rec.txt'},
                                        {'idx' : (0,1)}),
            'two2five' : ({'perf' : 'mspe_two2five_acc_prec_rec.txt'},
                                        {'idx' : (0,1)}),
            'five2two' : ({'perf' : 'mspe_five2two_acc_prec_rec.txt'},
                                                    {'idx' : (0,1)}),
            'null_five' : ({'perf' : 'null_perf_5.txt'},
                                        {'idx' : (1,1)}),
            'null_two' : ({'perf' : 'null_perf_2.txt'},
                                        {'idx' : (1,1)}),
            'null_two2five' : ({'perf' : 'null_two2five.txt'},
                                        {'idx' : (1,1)}),
            'null_five2two' : ({'perf' : 'null_five2two.txt'},
                                        {'idx' : (1,1)})         }
'''

fnames_dict = {'five' : ({'perf' : 'mspe5_searched_scores.txt'},
                            {'idx' : (1,1)}),
                'two' : ({'perf' : 'mspe2_searched_scores.txt'},
                                            {'idx' : (0,1)}),
                'null_five' : ({'perf' : 'null_perf_5.txt'},
                                            {'idx' : (1,1)}),
                'null_two' : ({'perf' : 'null_perf_2.txt'},
                                            {'idx' : (1,1)}),
                'two2five' : ({'perf' : 'mspe25_cat_searched_scores.txt'},
                                            {'idx' : (0,1)}),
                'five2two' : ({'perf' : 'mspe52_cat_searched_scores.txt'},
                                                        {'idx' : (0,1)}),
                'null_two2five' : ({'perf' : 'null_two2five.txt'},
                                            {'idx' : (1,1)}),
                'null_five2two' : ({'perf' : 'null_five2two.txt'},
                                            {'idx' : (1,1)})
                            }

perf_5 = my_load(fnames_dict['five'][0]['perf'])
acc_5 = np.asarray([perf_5[i][0] for i in range(len(perf_5))]) * 100
cm5 =  np.asarray([perf_5[i][1] for i in range(len(perf_5))]).mean(0)
rec_5 = np.asarray([perf_5[i][2] for i in range(len(perf_5))])
prec_5 = np.asarray([perf_5[i][3] for i in range(len(perf_5))])
f1_class5 = np.asarray([perf_5[i][4] for i in range(len(perf_5))]) * 100
#TWO
perf_2 = my_load(fnames_dict['two'][0]['perf'])
acc_2 = np.asarray([perf_2[i][0] for i in range(len(perf_2))]) * 100
cm2 =  np.asarray([perf_2[i][1] for i in range(len(perf_2))]).mean(0)
rec_2 = np.asarray([perf_2[i][2] for i in range(len(perf_2))])
prec_2 = np.asarray([perf_2[i][2] for i in range(len(perf_2))])
f1_class2 = np.asarray([perf_2[i][4] for i in range(len(perf_2))]) * 100
#TWO2FIVE
perf_two2five = my_load(fnames_dict['two2five'][0]['perf'])
acc_two2five = np.asarray([perf_two2five[i][0] for i in range(len(perf_2))]) * 100
f1_two2five = np.asarray([perf_two2five[i][4] for i in range(len(perf_two2five))]) * 100
#FIVE2TWO
perf_five2two = my_load(fnames_dict['five2two'][0]['perf'])
acc_five2two = np.asarray([perf_five2two[i][0] for i in range(len(perf_2))]) * 100
f1_five2two = np.asarray([perf_five2two[i][4] for i in range(len(perf_five2two))]) * 100

#NULL PERFORMANCE
null_perf_5 = my_load(fnames_dict['null_five'][0]['perf'])
null_acc_5 = np.asarray([np.asarray([null_perf_5[iter][f][0] for f in range(20)]) * 100 for iter in range(100) ])
null_perf_2 = my_load(fnames_dict['null_two'][0]['perf'])
null_acc_2 = np.asarray([np.asarray([null_perf_2[iter][f][0] for f in range(20)]) * 100 for iter in range(100) ])

null_perf_52 = my_load(fnames_dict['null_five2two'][0]['perf'])
null_acc_52 = np.asarray([np.asarray([null_perf_52[iter][f][0] for f in range(20)]) * 100 for iter in range(100) ])
null_perf_25 = my_load(fnames_dict['null_two2five'][0]['perf'])
null_acc_25 = np.asarray([np.asarray([null_perf_25[iter][f][0] for f in range(20)]) * 100 for iter in range(100) ])
null_f1_52= np.asarray([np.asarray([null_perf_52[iter][f][4] for f in range(20)]) * 100 for iter in range(100) ])


#plot per class
def get_pands(d, time):
    df = pd.DataFrame(d)
    if d.ndim == 2:
        df = df.rename(columns={0:'NREM', 1:'REM', 2:'WAKE'})
    elif d.ndim ==1:
        df = df.rename(columns={0:'acc'})
    df['time'] = [time] * len(d)
    return df

f1_5 = get_pands(f1_class5, time='week5')
f1_2 = get_pands(f1_class2, time='week2')
acc_5 = get_pands(acc_5, time='week5')
acc_2 = get_pands(acc_2, time='week2')

f1_25 = get_pands(f1_two2five, time='week5')
f1_52 = get_pands(f1_five2two, time='week2')
acc_25 = get_pands(acc_two2five, time='week5')
acc_52 = get_pands(acc_five2two, time='week2')

acc_cat = pd.concat([acc_2, acc_5])
f1_cat = pd.concat([f1_2, f1_5])
f1_cat_melt = pd.melt(f1_cat, id_vars= 'time')

acc_cross_cat =pd.concat([acc_52, acc_25])
f1_cross_cat = pd.concat([f1_52, f1_25])
f1_cross_cat_melt = pd.melt(f1_cross_cat, id_vars= 'time')



#PLOT WITHIN
def plot_mygrid(data2, data5, null2, null5, df, myxticklab ):
    fig = plt.figure(figsize=(4,5))
    gs1 = gridspec.GridSpec(2, 8, figure=fig)
    gs1.update(right=0.75, wspace=0.1)
    ax1 = plt.subplot(gs1[0, :3])
    ax2 = plt.subplot(gs1[0, 3:4])
    ax3= plt.subplot(gs1[0, 4:7])
    ax4= plt.subplot(gs1[0, 7:8])
    ax5= plt.subplot(gs1[1, :])

    sns.barplot(x='time', y = 'acc', data=data2,  ax=ax1, estimator=mean, ci=95, color='grey', \
                edgecolor='black')
    ax1.axhline(y=mean(null2.ravel()), linestyle='--', color='red', alpha=0.5)
    ax2.hist(null2.ravel(), bins=20, orientation='horizontal', color='darkgrey', edgecolor='black')

    sns.barplot(x='time', y = 'acc', data=data5,  ax=ax3, estimator=mean, ci=95, color='grey', \
                edgecolor='black')
    ax3.axhline(y=mean(null5.ravel()), linestyle='--', color='red', alpha=0.5)
    ax4.hist(null5.ravel(), orientation='horizontal', bins=12, color='darkgrey', edgecolor='black')

    sns.barplot(x='time', y = 'value', hue='variable', data=df,  ax=ax5, estimator=mean,
                ci=95, edgecolor='black')

    ax1.set(ylim=[0,90], xlabel='',  ylabel='Accuracy [%]', xticklabels=[],  xticks=[], \
            yticklabels=range(0,90,10),  yticks=range(0,90,10))
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax2.set(ylim=[0,90],  xlabel='', yticklabels=[],  xticks=[],   yticks=range(0,90,10))
    ax3.set(ylim=[0,90], xlabel='', ylabel='', xticks=[], xticklabels=[], \
            yticklabels=[],  yticks=range(0,90,10))
    ax3.yaxis.grid(color='gray', linestyle='dashed')
    ax4.set(ylim=[0,90],  xlabel='', yticklabels=[], xticks=[],  yticks=range(0,90,10))
    ax5.set_axisbelow(True)
    ax5.yaxis.grid(color='gray', linestyle='dashed')
    ax5.set(ylim=[0,90], ylabel='F1 [%]', xlabel='', xticks=[0,1], yticklabels=range(0,90,10),  yticks=range(0,90,10),\
            xticklabels=myxticklab)
    ax5.legend(prop=dict(size=12))
    legend_elements = [Line2D([0], [0], color='red', lw=2, alpha=0.5, label='Chance', linestyle='--')]
    #ax1.legend(handles=legend_elements, loc='best', prop={'size': 8})
    ax4.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.setp(ax5.get_legend().get_texts(), fontsize='8')
    ax5.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

#plot within
plot_mygrid(acc_2, acc_5, null_acc_2, null_acc_5, f1_cat_melt, myxticklab=['week2->week2', 'week5->week5'])
#plot cross
plot_mygrid(acc_52, acc_25, null_acc_52, null_acc_25, f1_cross_cat_melt,  myxticklab=['week5->week2', 'week2->week5'])








fig = plt.figure(figsize=(15, 4))
outer = gridspec.GridSpec(2, 4, wspace=0.1, hspace=0.8)

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
#plot accuracy within
df = pd.DataFrame({ 'two' : acc_2, 'five' : acc_5})
df = df.reindex(['two', 'five'], axis=1)
dfm = df.melt()
fig, ax = plt.subplots(1,1, figsize=(6, 6))
#sns.boxplot(x='variable', y = 'value', data=dfm,  ax=ax, width=0.3)
sns.barplot(x='variable', y = 'value', data=dfm,  ax=ax, estimator=mean, ci=95)
#sns.swarmplot(x='variable', y = 'value', data=dfm,  ax=ax, color ='black')
ax.set( ylabel='Accuracy [%]', xlabel='', xticks=[0,1], ylim= [50, 75],\
        xticklabels=['Week 2', 'Week 5'])

matplotlib.rcParams.update({'font.size': 15})

#U, pval = mann(df['two'], df['five'])


# week2 and five2two
df = pd.DataFrame({ 'two' : acc_2, 'five2two' : acc_five2two})
df = df.reindex(['two', 'five2two'], axis=1)
dfm = df.melt()
fig, ax = plt.subplots(1,1, figsize=(6, 6))
#sns.boxplot(x='variable', y = 'value', data=dfm,  ax=ax, width=0.3)
sns.barplot(x='variable', y = 'value', data=dfm,  ax=ax, estimator=mean, ci=95)
#sns.swarmplot(x='variable', y = 'value', data=dfm,  ax=ax, color ='black')
ax.set( ylabel='Accuracy [%]', xlabel='', xticks=[0,1], ylim= [50, 75],\
        xticklabels=['two',  'five2two'])

matplotlib.rcParams.update({'font.size': 15})

U, pval = mann(df['two'], df['five2two'])
print pval


# week5 and two2five
df = pd.DataFrame({ 'five' : acc_5, 'two2five' : acc_two2five})
df = df.reindex(['five', 'two2five'], axis=1)
dfm = df.melt()
fig, ax = plt.subplots(1,1, figsize=(6, 6))
#sns.boxplot(x='variable', y = 'value', data=dfm,  ax=ax, width=0.3)
sns.barplot(x='variable', y = 'value', data=dfm,  ax=ax, estimator=mean, ci=95)
#sns.swarmplot(x='variable', y = 'value', data=dfm,  ax=ax, color ='black')
ax.set( ylabel='Accuracy [%]', xlabel='', xticks=[0,1], ylim= [50, 75],\
        xticklabels=['five', 'two2five'])

matplotlib.rcParams.update({'font.size': 15})

U, pval = mann(df['five'], df['two2five'])
print pval

#plot xonfusion matrix for week 2 and week 5
plot_confusion_matrix(cm2, ['NREM', 'REM', 'WAKE'], title='', normalize=True)
plot_confusion_matrix(cm5, ['NREM', 'REM', 'WAKE'], title='', normalize=True)


#F1 scoresce stat
df = pd.concat([acc_25, acc_52])
U, pval = mann(df['REM'][df['time']=='week2'], \
        df['REM'][df['time']=='week5'])
print pval

# Acc scores stat
df = pd.concat([acc_25, acc_52])
U, pval = mann(df['acc'][df['time']=='week2'], \
        df['acc'][df['time']=='week5'])
print pval
