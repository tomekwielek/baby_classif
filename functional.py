import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mne import io
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from collections import Counter
from sklearn.gaussian_process import GaussianProcessClassifier as GP
from sklearn.gaussian_process.kernels import RBF
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

from config import (myload, paths, report, raw_path)

def map_stag_with_raw(fnames, s, sufx): #s=excel sheet e.g: St_Pr_corrected_27min
    '''get raw file name that corr. with stag name'''
    m = dict()
    for s_ in s.columns:
        m[s_] = [r_ for r_ in fnames if  np.logical_and(r_.startswith(s_[:5]), r_.endswith(sufx))]
    return m

def select_class_to_classif(pe, stag, sel_idxs):
    '''select relevant classes, type(sel_idxs) = list'''
    stag_sub = []
    pe_sub = []
    for i in range(len(stag)):
        if len(sel_idxs) == 2:
            idx_select = np.where(( stag[i]['numeric'] == sel_idxs[0]) |
                            (stag[i]['numeric'] == sel_idxs[1]) )[0]
        elif len(sel_idxs) == 3:
            idx_select = np.where(( stag[i]['numeric'] == sel_idxs[0]) |
                            (stag[i]['numeric'] == sel_idxs[1]) |
                            (stag[i]['numeric'] == sel_idxs[2]) )[0]
        else:
            raise NotImplementedError()
        stag_sub.append(stag[i].iloc[idx_select])
        pe_sub.append(pe[i][..., idx_select])
    return pe_sub, stag_sub

def merge_stages(stag, mapper):
    for i in range(len(stag)):
        stag[i]['numeric'].replace(mapper, inplace=True)
    return stag

def cat_pe_stag(pe, stag):
    '''concatenate pe and stag'''
    X = np.concatenate(pe,-1)
    X = np.vstack(X) if X.ndim > 2 else X
    X = X.transpose()
    stag_ = [stag[i].iloc[:,1].values for i in range(len(stag))]
    y = np.concatenate(stag_, 0)
    return X, y

def my_oversample(X, y):
    '''oversampling '''
    #X_resampled, y_resampled = ros.fit_sample(X, y)
    X_resampled, y_resampled = SMOTE(random_state=0).fit_sample(X, y)
    return X_resampled, y_resampled

def vis_PCA(X,y, sel_idxs):
    '''Visualize in 2 dimension after PCA'''
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, whiten=False)
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    if len(sel_idxs) == 2:
        plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class N', c='blue')
        plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class R', c='green', alpha=0.5)
    elif len(sel_idxs) == 3:
        plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class N', c='blue')
        plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class R', c='green', alpha=0.5)
        plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class W', c='red', alpha=0.5)
    else: raise NotImplementedError()
    plt.legend()
    plt.show()


def plot_pe(pe, stag, axes=None, mspe=False):
    '''plot pe for single subject'''
    from config import chs_incl
    if all(axes == None):
        fig, axes = plt.subplots(2,1)
    if mspe:
        ch_names = ['av_channs_taus'] # if mspe is ploted channels and taus are averaged
        pe = pe.mean(0)
    else:
        ch_names = chs_incl
    pe = pe.transpose()
    df = pd.DataFrame(pe, index=range(pe.shape[0]), columns=ch_names)
    df.plot(ax = axes[0])
    axes[0].set_ylabel('Permutation Entropy')
    time = range(len(stag))
    axes[1].plot(stag['numeric'].values, 'r*', label = 'ground truth (Scholle)')
    axes[1].set_xlim(0,len(stag))
    axes[1].set_ylim(0.8,3.2)
    axes[1].set_yticks([1,2,3])
    axes[1].set_yticklabels(['N','R','W'], fontsize='large')
    axes[1].legend()

def get_PCA_coef(pe,stag):
    X = np.concatenate([pe[i].transpose() for i in range(len(pe))], 0)
    sc = StandardScaler()
    pca = PCA(n_components=10)
    X = sc.fit_transform(X)
    pca.fit(X)
    coef = pca.components_
    #explained variance
    pca.explained_variance_ratio_.cumsum()

def vis_clf_probas(pe, stag):
    '''
    adapted from http://scikit-learn.org/stable/auto_examples/classification/plot_classification
    _probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py
    Get visualization of SVM decision boundries. Full dataset included. PCA to reduce dimensionality
    (only 2 dim plotable),
    '''
    X = np.concatenate([pe[i].transpose() for i in range(len(pe))], 0)
    y = np.concatenate([stag[i]['numeric'] for i in range(len(stag))], 0)

    X = PCA(n_components=2).fit_transform(X)

    xx = np.linspace(-0.5, 0.6, 100)
    yy = np.linspace(-0.2, 0.4, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    clf = make_pipeline(StandardScaler(), svm.SVC(C=1, kernel='linear',
                        probability=True))

    clf.fit(X,y)
    y_pred = clf.predict(X)
    probas = clf.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    classes = np.unique(y_pred)
    classes = [int(i) for i in classes]

    fig, axes = plt.subplots(1, n_classes, figsize = [11,6])
    #plt.suptitle('SVM classification and decision boundary')
    mapper = {1. : 'k', 2.: 'g', 3.: 'gold'}
    mapper_stag_name = {1. : 'NREM', 2.: 'REM', 3.: 'WAKE'}
    for i, k in enumerate(classes):
        axes[i].set_title('%s' % mapper_stag_name[k])
        #if i == 0:
            #axes[i].set_ylabel('PCA1')
            #axes[i].set_xlabel('PCA2')
        imshow_handle = axes[i].imshow(probas[:, i].reshape((100, 100)), cmap='coolwarm',
                                   extent=(-0.5, 0.6, -0.2, 0.4), origin='lower', vmin=0, vmax=1)
        axes[i].set_xticks(())
        axes[i].set_yticks(())
        idx = (y_pred == k)
        axes[i].autoscale(False)
        if idx.any():
            #axes[i].scatter(X[idx, 0], X[idx, 1], marker='o', c='k', alpha=0.5)
            axes[i].scatter(X[idx, 0], X[idx, 1], c=[mapper[i] for i in y[idx]], alpha=.5, s=5)
    ax = plt.axes([0.15, 0.24, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    markers = [plt.Line2D([0,0],[0,0],color=c, marker='o', linestyle='') for c in mapper.values()]
    ax2 = plt.axes([0.85, 0.65, 0.05, 0.15])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor('white')
    plt.legend(markers, mapper_stag_name.values(), loc=2, numpoints=3, frameon=False)

    plt.show()

def classify_lso(pe, stag, setup, myshow=True, oversample=False, mysave=False):
    '''classify using leave_sbj_out: single sbj out, optionally oversample'''
    #clf = make_pipeline(StandardScaler(), LogisticRegression(C = 1,
    #                                                        solver='liblinear',
    #                                                        class_weight=None,
    #                                                        multi_class='ovr'))
    clf = make_pipeline(MinMaxScaler(), svm.SVC(C=1, kernel='linear',
                        probability=False))
    no_sbjs = len(stag)
    kf = KFold(n_splits = no_sbjs)
    scores, predicts, f1s, accs, features, names, f1s_each =  ([] for i in range(7))
    check_mspe = pe[0][0].shape[0] > 11 #check if pe (then #chans=11) or mspe, for plotting only
    for train_idx, test_idx in kf.split(pe, stag):
        X = np.concatenate([pe[i].transpose() for i in train_idx], 0)
        y = np.concatenate([stag[i]['numeric'] for i in train_idx], 0)
        if oversample:
            X, y = RandomOverSampler(random_state=123).fit_sample(X, y)

        X_test = np.concatenate([pe[i].transpose() for i in test_idx], 0)
        y_test =  np.concatenate([stag[i]['numeric'] for i in test_idx], 0)
        clf.fit(X, y)
        #features.append(feature_selection(X,y))
        sbj_id_ = stag[test_idx[0]].columns[0][:3]
        pred_ = clf.predict(X_test)
        score_ = clf.score(X_test, y_test)

        #plot clf probabilitites
        #vis_clf_probas(X_test, y_test)
        #average f1, (weighted-is preferable if class imbalance),ignore scores for labels absent in pred_
        f1_ = f1_score(pred_, y_test, average = 'weighted', labels=np.unique(pred_))
        f1_each_ = f1_score(pred_, y_test, average=None, labels=[1., 2., 3.])
        acc_ = accuracy_score(pred_, y_test, normalize=True)
        predicts.append((pred_, y_test))
        scores.append(score_)
        f1s.append(f1_); accs.append(acc_); f1s_each.append(f1_each_)
        names.append(sbj_id_)
        print test_idx
        #if check_mspe: #for mspe average channels and taus
        #    pe = [pe[i].mean(0) for i in range(len(pe))]
        if myshow:
            fig, axes = plt.subplots(3,1, sharex = True, figsize = [12,7])
            plot_pe(pe[test_idx[0]], stag[test_idx[0]], axes = axes[:2], mspe=check_mspe)

            axes[2].plot(pred_, 'bo', label = 'prediction')
            axes[2].set_xlim(0,len(pred_))
            axes[2].set_ylim(0.8,3.2)
            axes[2].set_yticks([1,2,3])
            axes[2].set_yticklabels(['N','R','W'], fontsize='large')
            axes[2].legend()
            times = range(len(pred_))
            axes[2].set_xticks(times)
            axes[2].set_xticklabels('time [30s]', fontsize='large')
            f1_str = 'f1=' + str(round(f1_,2))
            #acc_str = 'Accuracy=' + str(round(acc_,3))
            fig.text(0.05, 0.95, [setup, f1_str], size = 22)
            plt.show()
            report.add_figs_to_section(fig, captions='Sbj '+ str(sbj_id_),
                                       section= setup)
        if mysave:
            from config import mysave
            typ_name = 'pred'
            sbj_save = stag[test_idx[0]].columns[0][:5]
            mysave(var = pred_, typ=typ_name, sbj=sbj_save)
    return predicts, scores, report, f1s, names, f1s_each

def subject_h0(pe, stag, no_perm):
    '''
    Based on Stelzer(2013). Get single subject chance accuracy estimate by permuting
    stag labeling 'no_perm' times.
    '''
    clf = make_pipeline(MinMaxScaler(), svm.SVC(C=1, kernel='linear',probability=False))
    no_sbjs = len(stag)
    kf = KFold(n_splits = no_sbjs/2)
    store_sbjs_h0 = []
    for perm_idx in range(no_perm):
        #shuffle full y, alternativly shuffle within fold
        #stag_shuff = [stag[i].sample(frac=1).reset_index(inplace=False, drop=True) for i in range(len(stag))]
        h0 = []
        #for train_idx, test_idx in kf.split(pe, stag_shuff):
        for train_idx, test_idx in kf.split(pe, stag):
            X = np.concatenate([pe[i].transpose() for i in train_idx], 0)

            #y = np.concatenate([stag_shuff[i]['numeric'] for i in train_idx], 0)
            y = np.concatenate([stag[i]['numeric'] for i in train_idx], 0)
            np.random.shuffle(y)
            X_test = np.concatenate([pe[i].transpose() for i in test_idx], 0)
            #y_test =  np.concatenate([stag_shuff[i]['numeric'] for i in test_idx], 0)
            y_test =  np.concatenate([stag[i]['numeric'] for i in test_idx], 0)
            np.random.shuffle(y_test)
            clf.fit(X, y)
            pred_ = clf.predict(X_test)
            #scoring same as for real data, see upper
            f1_ = f1_score(pred_, y_test, average = 'weighted', labels=np.unique(pred_))
            acc_ = accuracy_score(pred_, y_test, normalize=True)
            h0.append(acc_)
        store_sbjs_h0.append(h0)
        print perm_idx
    return np.asarray(store_sbjs_h0)
'''
Based on Stelzer(2013). Get group chance level by randomly selecting one accuracy per subjects
and averging. Repeat 'times' times. p value is calculated by r/times where r is count
of accuracies grater than the actual accuracy. Notice correct formula acc. [North, 2002] is:
(r+1)/(times+1)
'''
def group_h0(pe, stag, my_score, times=10e4):
    times = int(times)
    h0_sbj = subject_h0(pe, stag, no_perm=500)
    no_sbj = h0_sbj.shape[1]
    h0_gr = np.zeros([times, no_sbj])
    for i in range(times):
        h0_gr[i, :] = [np.random.choice(h0_sbj[sbj_id,:]) for sbj_id in range(no_sbj)]
        print i
    h0_av = h0_gr.mean(1)

    my_score = np.asarray(my_score).mean(0)
    r = h0_av > my_score
    p = r.sum() / float(len(h0_gr))
    return h0_gr, p

def classify_generalize_time(pe1, stag1, pe2, stag2, setup, show=True):
    '''classify using leave_sbj_out; test on single left out as well as
    single left out from the second recording time (fnames2)'''
    '''
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('scl', MinMaxScaler()),
		           ('clf', svm.SVC())])

    grid_params = [{'clf__kernel': ['linear', 'rbf'],
		          'clf__C': [0.001, 0.1, 1, 10, 100],
                  'clf__gamma' : [0.1, 0.01, 0.001, 0.0001]}]

    from sklearn.grid_search import GridSearchCV
    gs = GridSearchCV(estimator=pipe,
			param_grid=grid_params,
			scoring='accuracy',
            cv=10)
    '''
    clf = make_pipeline(StandardScaler(), svm.SVC(C=1., kernel='linear',
                        probability=False))
    #clf = make_pipeline(StandardScaler(), LogisticRegression(C=1,
    #                                                        solver='liblinear',
    #                                                        class_weight=None,
    #                                                        multi_class='ovr'))
    #pe1, pe2, stag1, stag2 = align_t1_t2_data(pe1, pe2, stag1, stag2)

    #pe1, pe2, stag1, stag2 = (pe2, pe1, stag2, stag1) #train t1 test t2
    assert len(stag1) == len(stag2)
    no_sbjs = len(stag1)
    kf = KFold(n_splits = no_sbjs)

    predicts1, predicts2, f1_1, f1_2, f1_1_NREM, names1, names2 =  ([] for i in range(7))

    for train_idx, test_idx in kf.split(pe1, stag1):
        sbj_id_1 = stag1[test_idx[0]].columns[0][:3]
        sbj_id_2 = stag2[test_idx[0]].columns[0][:3]

        X = np.concatenate([pe2[i].transpose() for i in train_idx], 0)
        y = np.concatenate([stag2[i]['numeric'] for i in train_idx], 0)

        X_test1 = np.concatenate([pe1[i].transpose() for i in test_idx], 0)
        y_test1 =  np.concatenate([stag1[i]['numeric'] for i in test_idx], 0)

        X_test2 = np.concatenate([pe2[i].transpose() for i in test_idx], 0)
        y_test2 =  np.concatenate([stag2[i]['numeric'] for i in test_idx], 0)

        clf.fit(X, y)
        #gs.fit(X, y)
        pred_1 = clf.predict(X_test1)
        pred_2 = clf.predict(X_test2)

        f1_1_ = f1_score(pred_1, y_test1, average = 'micro')
        f1_1_NREM_ = f1_score(pred_1, y_test1, labels=[1], average = 'micro')
        f1_2_ = f1_score(pred_2, y_test2, average = 'micro')

        predicts1.append((pred_1, y_test1))
        predicts2.append((pred_2, y_test2))

        f1_1.append(f1_1_)
        f1_2.append(f1_2_)
        f1_1_NREM.append(f1_1_NREM_)
        names1.append(sbj_id_1)
        names2.append(sbj_id_2)
        if show:
            assert stag1[test_idx[0]].columns[0][:3] == stag2[test_idx[0]].columns[0][:3]
            sbj_id_ = stag1[test_idx[0]].columns[0][:3]
            fig, axes = plt.subplots(6,1, sharex = True, figsize = [12,7])
            plot_pe(pe1[test_idx[0]], stag1[test_idx[0]], axes = axes[:2])
            axes[2].plot(pred_1, 'bo', label = 'prediction within')
            axes[2].set_xlim(0,len(pred_1))
            axes[2].set_ylim(0.8,3.2)
            axes[2].set_yticks([1,2,3])
            axes[2].set_yticklabels(['N','R','W'], fontsize='large')
            axes[2].legend()
            times = range(len(pred_1))
            axes[2].set_xticks(times)
            axes[2].set_xticklabels('time [30s]', fontsize='large')

            #plot predictin across time
            plot_pe(pe2[test_idx[0]], stag2[test_idx[0]], axes = axes[3:5])
            axes[5].plot(pred_2, 'bo', label = 'prediction across')
            axes[5].set_xlim(0,len(pred_2))
            axes[5].set_ylim(0.8,3.2)
            axes[5].set_yticks([1,2,3])
            axes[5].set_yticklabels(['N','R','W'], fontsize='large')
            axes[5].legend()
            times = range(len(pred_2))
            axes[5].set_xticks(times)
            axes[5].set_xticklabels('time [30s]', fontsize='large')
            fig.suptitle(sbj_id_)
            fig.text(0.05, 0.95, setup, size = 22)
            #report.add_figs_to_section(fig, captions='Sbj '+ str(sbj_id_),
            #                            section= setup)
    #print('Best accuracy: %.3f' % gs.best_score_)

    # Best params
    #print('\nBest params:\n', gs.best_params_)

    return report, predicts1, predicts2, f1_1, f1_2, f1_1_NREM, names1, names2


def align_t1_t2_data(pe1, pe2, stag1, stag2):
    '''
    not every
     subject appear during t1 and t2; include matches only
    '''
    #sort staging and pe simultaneously
    data1 = sorted(zip(pe1, stag1), key=lambda x: x[1].columns[0])
    data2 = sorted(zip(pe2, stag2), key=lambda x: x[1].columns[0])

    _, stag1_sorted = map(list, zip(*data1))
    _, stag2_sorted = map(list, zip(*data2))

    names1 = [stag1_sorted[i].columns[0][:3] for i in range(len(stag1_sorted))]
    names2 = [stag2_sorted[i].columns[0][:3] for i in range(len(stag2_sorted))]
    # get data that intersect
    data1 = [data1[i] for i in range(len(data1)) if names1[i] in names2]
    data2 = [data2[i] for i in range(len(data2)) if names2[i] in names1]
    assert [data2[i][1].columns[0][:3] for i in range(len(data2))] == \
        [data1[i][1].columns[0][:3] for i in range(len(data1))]
    pe1, stag1 = map(list, zip(*data1))
    pe2, stag2 = map(list, zip(*data2))
    return pe1, pe2, stag1, stag2

def load_single_append(path, fnames, typ):
    '''Load single subjects, drop nan from stag, make pe and stag of equall length
    (differences of 1 due to windowing in pe computaton) and append a list'''
    freqs = None
    stag_list, pe_list, n_list = ([] for i in range(3))
    #if '236_2' in fnames: #stag- and pe- length in '236_2' does not match TODO
    #    fnames.remove('236_2')
    for s in fnames:
        #omit if filepath is empty
        if not os.path.isfile(paths(typ=typ, sbj=s)):
            print 'Folder is empty'
            print s
            continue
        if typ == 'psd':
            stag, pe, freqs = myload(typ=typ, sbj=s) # for psd load freqs bins

        else:
            stag, pe = myload(typ=typ, sbj=s)
            #pe = pe[0:10, :] # single sbj has 11 channels, why? TODO
        if s in ['110_2']: #see annot by HL in excel; BL shorter
            stag = stag.iloc[:66]
        if s in ['236_2']:
            stag  = stag[1:] #see annot by HL in excel; BL shorter
        stag = stag.dropna()
        if len(stag) - pe.shape[-1] == 1:#pe can be shorter due to windowing
            stag = stag[:-1]
        else:
             NotImplementedError
        print s
        print len(stag)
        print pe.shape[1]
        assert len(stag) == pe.shape[-1]
        stag_list.append(stag)
        pe_list.append(pe)
        n_list.append(s)

    return pe_list, stag_list, n_list, freqs

def count_stag(stag):
    no_sbj = len(stag)
    c = zip(range(no_sbj),[Counter(stag[i]['numeric']).items() for i in range(no_sbj)])
    return c

def compare_pe(pe, stag):
    av_pe = dict()
    for s in sel_idxs: #average pe avross epochs for sleep stages
        av_pe[s] = [pe[i].transpose()[stag[i]['numeric']==s].mean() for i in range(len(pe))]
        av_pe['av'+str(s)] =  np.asarray(av_pe[s])[~np.isnan(av_pe[s])].mean()
    return av_pe

def feature_selection(X, y):
    X = StandardScaler().fit_transform(X)
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    return clf.feature_importances_

def my_rename_col(s):
    '''
    rename columns of excell sheet such that '-' => '_' (match with naming conventions
    of raw fnames)
    '''
    s_list_ = list(s.columns)
    s_new = [s_.replace('-', '_') for s_ in s_list_]
    mapper = dict(zip(s.columns, s_new))
    return s.rename(columns=mapper, copy=False)

def pandas_count_items(df, var_name, str_length):
    '''
    counter for pandas df; TODO
    '''
    u = [df[var_name].iloc[i][:str_length] for i in range(len(df))]
    c = Counter(u).items()
    return u, c

def order_channels(raw):
    '''
    for missing channels add zeros, reorder channels  across subjects
    '''
    from config import chs_incl
    import mne
    chs = raw.info['ch_names']
    missing = [ch for ch in chs_incl if ch not in chs]
    print 'MISSING CHANNELS: %s' %missing
    # add missing channels as zeros data (!!!)
    miss_data = np.zeros((len(missing), len(raw.times)))
    if len(missing) == 1: #single channel is missing
        miss_info = mne.create_info(missing, raw.info['sfreq'], ['eeg'])
        miss_raw = mne.io.RawArray(miss_data[0:], miss_info)
        raw.add_channels([miss_raw], force_update_info=True)
    elif len(missing) == 2: # two channels are missing
        miss_info1 = mne.create_info([missing[0]], raw.info['sfreq'], ['eeg'])
        miss_raw1 = mne.io.RawArray(miss_data[[0],:], miss_info1)
        miss_info2 = mne.create_info([missing[1]], raw.info['sfreq'], ['eeg'])
        miss_raw2 = mne.io.RawArray(miss_data[[1],:], miss_info2)
        raw.add_channels([miss_raw1], force_update_info=True)
        raw.add_channels([miss_raw2], force_update_info=True)

    raw_ordered = raw.reorder_channels(chs_incl)
    #raw.save(fname=raw_path+'test'+fn, overwrite=False)
    return raw_ordered, missing
