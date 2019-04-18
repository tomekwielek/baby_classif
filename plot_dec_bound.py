import os
import numpy as np
from config import myload, base_path, paths
from functional import (select_class_to_classif, cat_pe_stag, load_single_append,
                        count_stag, vis_clf_probas, write_pickle, read_pickle, plot_confusion_matrix,\
                        plot_acc_depending_scale, remove_20hz_artif)
from matplotlib import pyplot as plt
from read_raw import *
from mne import io
import pickle
from IPython.core.debugger import set_trace
from config import bad_sbjs_1, bad_sbjs_2
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
from sklearn.manifold import MDS, SpectralEmbedding
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders
sel_idxs = [1,2,3]
time = 2  # if 2 week2, elif 5 week5, elif 'cat' concatenate
n_folds = 2

setup = 'mspet1m3'
s = 5 #taus no

mspe1, mspe_stag1, mspe_names1, _ = load_single_append(path, fnames1, typ='mspet1m3')
mspe1_nf, _, _, _ = load_single_append(path, fnames1, typ='mspet1m3_nofilt')

mspe2, mspe_stag2, mspe_names2, _ = load_single_append(path, fnames2, typ='mspet1m3')
psd1, psd_stag1, psd_names1, freqs = load_single_append(path, fnames1, typ='psd')
psd2, psd_stag2, psd_names2, freqs = load_single_append(path, fnames2, typ='psd')

assert all([all(mspe_stag1[i] == psd_stag1[i]) for i in range(len(psd_stag1))])
assert all([all(mspe_stag2[i] == psd_stag2[i]) for i in range(len(psd_stag2))])
del (psd_stag1, psd_stag2)

mspe1, psd1, stag1,_ = remove_20hz_artif(mspe1, psd1, mspe_stag1, mspe_names1, freqs, bad_sbjs_1)
mspe2, psd2, stag2, _ = remove_20hz_artif(mspe2, psd2, mspe_stag2, mspe_names2, freqs, bad_sbjs_2)

mspe1, stag1, _ = select_class_to_classif(mspe1, stag1, sel_idxs=sel_idxs)
mspe2, stag2, _ = select_class_to_classif(mspe2, stag2, sel_idxs=sel_idxs)
psd1, stag1, _ = select_class_to_classif(psd1, stag1, sel_idxs=sel_idxs)
psd2, stag2, _ = select_class_to_classif(psd2, stag2, sel_idxs=sel_idxs)

mspe1_ = [ mspe1[i ][:s, ...] for i in range(len(mspe1)) ]  #use scale: 1, 2, 3, 4 only
mspe1 = [ mspe1_[i].reshape(-1, mspe1_[i].shape[-1]) for i in range(len(mspe1_)) ] # reshape

mspe2_ = [ mspe2[i ][:s, ...] for i in range(len(mspe2)) ] #use scale: 1, 2, 3, 4 only
mspe2 = [ mspe2_[i].reshape(-1, mspe2_[i].shape[-1]) for i in range(len(mspe2_)) ] #reshape

if setup == 'mspet1m3':
    if time == 2:
        data_pe, data_stag = mspe1, stag1
    elif time == 5:
        data_pe, data_stag = mspe2, stag2
    elif time == 'cat':
        data_pe = mspe1 + mspe2
        data_stag = stag1 + stag2

no_sbjs = len(data_stag)

kf = RepeatedKFold(n_splits=n_folds, n_repeats=1, random_state=11)

for _, out_idx in kf.split(range(no_sbjs)):
    # TEST data
    X_test = [data_pe[i] for i in range(len(data_pe)) if i in out_idx]
    X_test = np.hstack(X_test).T
    y_test = [data_stag[i] for i in range(len(data_stag)) if i in out_idx]
    y_test = np.vstack(y_test)[:,1].astype('int')
    #TRAIN data
    X_train = [data_pe[i] for i in range(len(data_pe)) if i not in out_idx]
    y_train = [data_stag[i] for i in range(len(data_stag)) if i not in out_idx]
    X_train = np.hstack(X_train)
    y_train= np.vstack(y_train)[:,1].astype('int')
    X_train = X_train.T

    #scale
    #X_train = StandardScaler().fit_transform(X_train)
    #X_test = StandardScaler().fit_transform(X_test)
    #resample
    no_samples =  dict([(1, 100), (2, 100), (3, 100)])
    rus = RandomUnderSampler(random_state=0, ratio=no_samples)
    rus.fit(X_train, y_train)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    rus = RandomUnderSampler(random_state=0, ratio=no_samples)
    rus.fit(X_test, y_test)
    X_test, y_test = rus.fit_resample(X_test, y_test)

    X_train_embedded = MDS(n_components=2).fit_transform(X_test)
    #X_train_embedded = TSNE(n_components=2).fit_transform(X_test)
    #X_train_embedded = PCA(n_components=2).fit_transform(X_test)

    model = ExtraTreesClassifier(40, n_jobs=-1).fit(X_train, y_train)
    #model = LogisticRegression().fit(X_train,y_train)
    y_predicted = model.predict(X_test)

    # create meshgrid
    resolution = 20 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_train_embedded[:,0]), np.max(X_train_embedded[:,0])
    X2d_ymin, X2d_ymax = np.min(X_train_embedded[:,1]), np.max(X_train_embedded[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    #plot
    colors_dict ={1: 'red', 2:'grey', 3:'blue'}
    colors = [colors_dict[y_test[i]] for i in range(len(y_test)) ]

    plt.contourf(xx, yy, voronoiBackground, cmap='RdBu', linewidths=2, alpha=0.5)
    plt.scatter(X_train_embedded[:,0], X_train_embedded[:,1], c=colors, s=23)
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('MDS 1')
    plt.xlabel('MDS 2')
    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Ellipse(xy=center, width=height + xdescent,
                                 height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    colors = list(colors_dict.values())
    texts = ['NREM', 'REM', 'WAKE']
    c = [ mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=colors[i], edgecolor="none" ) for i in range(len(texts))]
    plt.legend(c,texts,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., ncol=1,
            handler_map={mpatches.Circle: HandlerEllipse()}).get_frame()
    plt.show()
    plt.figure()



#PLot importnaces and corresponding scatter plots (channels as dimensions)

#LOAD IMPORTANCES
chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'EMG', 'HEOG_l', 'HEOG_r', 'VEOG']
data = read_pickle('H:\\BABY\\results\\scores\\importances\mspecat_including_imps.txt')
#PLOT av, sorted IMPORTANCES
imps = np.asarray([data[i][-1] for i in range(len(data))])
imps = imps.mean(1) #av scales
imps_sorted_idx = np.argsort(imps.mean(0))[::-1]
imps_df = pd.DataFrame(imps, columns=chs_incl)
imps_df = pd.melt(imps_df)
#pal = sns.color_palette('Blues_d', n_colors=len(chs_incl))
chs_ordered = np.asarray(chs_incl)[imps_sorted_idx]
colors  = ['grey' if c in ['F3', 'C3', 'O1', 'O2', 'C4', 'F4',] else 'lightgrey' for _, c in enumerate(chs_ordered)]
#color = [x for x, _ in sorted(zip(color,imps_sorted_idx), key=lambda x: x[1])]
sns.barplot( x='variable', y='value', data=imps_df, palette = colors, #palette=pal,
                order=chs_ordered)
ax = plt.gca()
ax.set_ylabel('Relative importance', fontsize=20)
ax.set_xlabel('')
ax.set_xticklabels(chs_ordered, rotation=45,fontsize=16)
plt.show()
'''
# GET FULL DATASET no cv
X = [data_pe[i] for i in range(len(data_pe)) ]
X = np.hstack(X).T
y = [data_stag[i] for i in range(len(data_stag))]
y = np.vstack(y)[:,1].astype('int')

#X = StandardScaler().fit_transform(X)
#X = StandardScaler().fit_transform(X)

#resample
no_samples =  dict([(1, 100), (2, 100), (3, 100)])
rus = RandomUnderSampler(random_state=0, ratio=no_samples)
rus.fit(X, y)
X, y = rus.sample(X, y)

#PLOT SCATTER FOR 3 BEST channels
fig_s, (axes) = plt.subplots(1,3)
colors_dict ={1: 'r', 2:'b', 3:'g'}
colors = [colors_dict[y[i]] for i in range(len(y)) ]
best_idx = imps_sorted_idx[:3]
X3d = X[:, best_idx]
for  i, ch_idx in enumerate([(0,2), (0,1), (1,2)]):
    xch= np.asarray(chs_incl)[imps_sorted_idx][ch_idx[0]]
    ych= np.asarray(chs_incl)[imps_sorted_idx][ch_idx[1]]
    axes[i].scatter(X3d[:,ch_idx[0]], X3d[:,ch_idx[1]], c=colors,  s=60, label=
            (xch, ych))
    axes[i].set(xlabel=xch, ylabel=ych)
'''








'''
ax.set_xlabel(chs_incl[best_idx[0]])
ax.set_ylabel(chs_incl[best_idx[1]])
ax.set_zlabel(chs_incl[best_idx[2]])
legend_elements = [Patch(facecolor=colors_dict.values()[0], edgecolor='b',
                         label='NREM'),
                Patch(facecolor=colors_dict.values()[1], edgecolor='b',
                                         label='REM'),
                Patch(facecolor=colors_dict.values()[2], edgecolor='b',
                                         label='WAKE')]
plt.legend(handles=legend_elements, loc='upper right')
plt.show()
'''
