import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import io

def map_stag_with_raw(raw_fnames, s): #s=sheet from excel file, e.g: St_Pr_corrected_27min
    '''get raw file name that corr. with stag name'''
    m = {s_ : [r_ for r_ in raw_fnames if np.logical_and(r_.startswith(s_[:5]), r_.endswith('ref100'))] for s_ in s.columns}
    return m

def select_stages_to_classif(pe, stag, sel_idxs):
    '''select relevant stages, type(sel_idxs) = list'''
    stag_sub = []; pe_sub = []
    for i in range(len(stag)):
        if len(sel_idxs) == 2:
            idx_select = np.where(( stag[i]['numeric'] == sel_idxs[0]) |
                            (stag[i]['numeric'] == sel_idxs[1]) )[0]
        elif len(sel_idxs) == 3:
            idx_select = np.where(( stag[i]['numeric'] == sel_idxs[0]) |
                            (stag[i]['numeric'] == sel_idxs[1]) |
                            (stag[i]['numeric'] == sel_idxs[2]) )[0]
        else: raise NotImplementedError()
        stag_sub.append(stag[i].iloc[idx_select])
        pe_sub.append(pe[i][..., idx_select])
        del idx_select
    return pe_sub, stag_sub

def merge_stages(stag, mapper):
    for i in range(len(stag)):
        stag[i].iloc[:,[0]].replace(mapper, inplace=True).astype(float)
    return stag

def cat_pe_stag(pe, stag):
    '''concatenate pe and stag'''
    X = np.concatenate(pe,-1)
    X = np.vstack(X) if X.ndim > 2 else X
    X = X.transpose()
    stag_ = [stag[i].iloc[:,1].values for i in range(len(stag))]
    y = np.concatenate(stag_, 0)
    return X, y
'''oversampling '''
def my_oversample(X, y):
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE, ADASYN
    #X_resampled, y_resampled = ros.fit_sample(X, y)
    X_resampled, y_resampled = SMOTE(random_state=0).fit_sample(X, y)
    return X_resampled, y_resampled
'''Visualize in 2 dimension after PCA'''
def vis_PCA(X,y, sel_idxs):
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, RobustScaler
    X_norm = StandardScaler().fit_transform(X)

    pca = sklearnPCA(n_components=2, whiten=False)
    #pca = KMeans(n_clusters=3)
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
'''Visualize in 2 dimension after ICA'''
def vis_ICA(X, y, sel_idxs):
    from sklearn.decomposition import FastICA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    X_norm = StandardScaler().fit_transform(X)
    n_components = len(sel_idxs)
    ica = FastICA(n_components=n_components, whiten = False)
    S_ica_ = ica.fit(X).transform(X)  # Estimate the sources
    S_ica_ /= S_ica_.std(axis=0)
    transformed = pd.DataFrame(S_ica_)

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

'''select single freq. bin TODO: check if transpose() correct!'''
def single_freq(power, f):
    power_single = [power[i][...,f].transpose() for i in range(len(power))]
    return power_single

'''classify, NO leave_sbj_out: X,y are pooled across subjects'''
def classify(X, y):
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler
    #clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    #cv = StratifiedKFold(n_splits = 5, shuffle= True)
    cv = TimeSeriesSplit(n_splits=5)
    scores=[]; predicts=[]; y_test=[]
    for train_idx, test_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        pred_ = clf.predict(X[test_idx])
        predicts.append(pred_)
        y_test.append(y[test_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))
    plt.plot(np.hstack(y_test))
    plt.plot(np.hstack(predicts))
    return predicts, scores
'''plot pe for single subject'''
def plot_pe(pe, stag, axes=None):
    from config import raw_path
    if all(axes == None):
        fig, axes = plt.subplots(2,1)
    ch_names = io.read_raw_edf(raw_path + '104_2_correctFilter_2heogs_ref100.edf',
            preload=False).info['ch_names'][:-1]
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

def plot_mspe(pe, stag):#, pred):
    fig, axes = plt.subplots(2,1)
    #ch_names = raw.info['ch_names']
    pe = pe.mean(1) #average channels
    for i in range(4):
        axes[0].plot(pe[i,...])

    axes[1].plot(stag['numeric'])
    plt.show()

'''classify using leave_sbj_out: single sbj out, optionally oversample'''
def classify_lso(pe, stag, setup, oversample=False, show=True):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier as rf
    from sklearn.dummy import DummyClassifier
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE, ADASYN
    from sklearn import svm
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import f1_score, accuracy_score
    from config import report
    from functional import feature_selection
    from sklearn.model_selection import LeaveOneGroupOut
    clf = make_pipeline(StandardScaler(), LogisticRegression(C = 1,
                                                            solver='liblinear',
                                                            class_weight=None,
                                                            multi_class='ovr'))
    clf = make_pipeline(StandardScaler(), rf())
    #clf = make_pipeline(StandardScaler(), DummyClassifier('most_frequent'))
    no_sbjs = len(stag)
    kf = KFold(n_splits = no_sbjs)
    #logo = LeaveOneGroupOut()
    #create grouping index for logo
    #lens = [pe[i].shape[1] for i in range(len(pe))]
    #idxs = np.arange(len(pe))
    #groups = [[idxs[i]]*lens[i] for i in range(len(pe))]

    #storers for scores
    scores = []; predicts = []; f1 = []; acc =[]; features=[]
    print scores 
    for train_idx, test_idx in kf.split(pe, stag):
        X = np.concatenate([pe[i].transpose() for i in train_idx], 0)
        y = np.concatenate([stag[i]['numeric'] for i in train_idx], 0)
        if oversample:
            X, y = RandomOverSampler(random_state=123).fit_sample(X, y)
        X_test = np.concatenate([pe[i].transpose() for i in test_idx], 0)
        y_test =  np.concatenate([stag[i]['numeric'] for i in test_idx], 0)
        if oversample:
            X_test, y_test = RandomOverSampler(random_state=123).fit_sample(X_test, y_test)
        clf.fit(X, y)
        features.append(feature_selection(X,y))
        sbj_id_ = stag[test_idx[0]].columns[0][:3]
        pred_ = clf.predict(X_test)
        score_ = clf.score(X_test, y_test)
        f1_ = f1_score(pred_, y_test, average = 'weighted')
        acc_ = accuracy_score(pred_, y_test, normalize=True)
        predicts.append((pred_, y_test))
        scores.append(score_)
        f1.append(f1_); acc.append(acc_)
        if show:
            fig, axes = plt.subplots(3,1, sharex = True, figsize = [12,7])
            plot_pe(pe[test_idx[0]], stag[test_idx[0]], axes = axes[:2])
            axes[2].plot(pred_, 'bo', label = 'prediction')
            axes[2].set_xlim(0,len(pred_))
            axes[2].set_ylim(0.8,3.2)
            axes[2].set_yticks([1,2,3])
            axes[2].set_yticklabels(['N','R','W'], fontsize='large')
            axes[2].legend()
            times = range(len(pred_))
            axes[2].set_xticks(times)
            axes[2].set_xticklabels('time [30s]', fontsize='large')
            #fig.suptitle(str(test_idx))
            f1_str = 'f1=' + str(round(f1_,2))
            acc_str = 'Accuracy=' + str(round(acc_,3))
            fig.text(0.05, 0.95, [setup, acc_str], size = 22)
            report.add_figs_to_section(fig, captions='Sbj '+ str(sbj_id_),
                                        section= setup)
    return predicts, scores, report, features

'''classify using leave_sbj_out; test on single left out as well as
single left out from the second recording time (fnames2)'''
def classify_generalize_time(pe1, stag1, pe2, stag2, setup, show=True):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from config import report
    from functional import align_t1_t2_data
    clf = make_pipeline(StandardScaler(), LogisticRegression(C = 1,
                                                            solver='liblinear',
                                                            class_weight=None,
                                                            multi_class='ovr'))
    pe1, pe2, stag1, stag2 = align_t1_t2_data(pe1, pe2, stag1, stag2)

    pe1, pe2, stag1, stag2 = (pe2, pe1, stag2, stag1) #train t2 test t1
    assert len(stag1) == len(stag2)
    no_sbjs = len(stag1)
    kf = KFold(n_splits = no_sbjs)
    predicts1 = []; predicts2 = []
    f1_1 = []; f1_2 = []

    for train_idx, test_idx in kf.split(pe1, stag1):
        X = np.concatenate([pe1[i].transpose() for i in train_idx], 0)
        y = np.concatenate([stag1[i]['numeric'] for i in train_idx], 0)

        X_test1 = np.concatenate([pe1[i].transpose() for i in test_idx], 0)
        y_test1 =  np.concatenate([stag1[i]['numeric'] for i in test_idx], 0)

        X_test2 = np.concatenate([pe2[i].transpose() for i in test_idx], 0)
        y_test2 =  np.concatenate([stag2[i]['numeric'] for i in test_idx], 0)

        clf.fit(X, y)
        pred_1 = clf.predict(X_test1)
        pred_2 = clf.predict(X_test2)

        f1_1_ = f1_score(pred_1, y_test1, average = 'macro')
        f1_2_ = f1_score(pred_2, y_test2, average = 'macro')

        predicts1.append((pred_1, y_test1))
        predicts2.append((pred_2, y_test2))

        f1_1.append(f1_1_)
        f1_2.append(f1_2_)
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
            #fig.suptitle(str(test_idx))
            fig.suptitle(sbj_id_)
            fig.text(0.05, 0.95, setup, size = 22)
            report.add_figs_to_section(fig, captions='Sbj '+ str(sbj_id_),
                                        section= setup)
                        #report.save(save_report, overwrite=True)
    return report

'''
not every subject appear during t1 and t2; include matches only
'''
def align_t1_t2_data(pe1, pe2, stag1, stag2):
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


'''Load single subjects, drop nan from stag, make pe and stag of equall length
(differences of 1 due to windowing in pe computaton) and append a list'''
def load_single_append(path, fnames, typ):
    from config import myload, paths
    import os
    stag_list = []; pe_list = []; s_list = []
    #stag- and pe- length in '236_2' does not match TODO
    if '236_2' in fnames:
        fnames.remove('236_2')
    for s in fnames:
        #omit if filepath is empty
        if not os.path.isfile(paths(typ=typ, sbj=s)):
            print 'Folder is empty'
            print s
            continue
        stag, pe = myload(typ=typ, sbj=s)
        if s in ['110_2']:#see annot by HL in excel; BL shorter
            stag = stag.iloc[:66]
        stag = stag.dropna()
        if len(stag) - pe.shape[1] == 1:#pe can be shorter due to windowing
            stag = stag[:-1]
        else:
             NotImplementedError
        print s
        print len(stag)
        print pe.shape[1]
        assert len(stag) == pe.shape[1]
        stag_list.append(stag)
        pe_list.append(pe)
        s_list.append(s)

    return pe_list, stag_list, s_list

def count_stag(stag):
    from collections import Counter
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
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
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
    from collections import Counter
    u = [df[var_name].iloc[i][:str_length] for i in range(len(df))]
    c = Counter(u).items()
    return u, c
