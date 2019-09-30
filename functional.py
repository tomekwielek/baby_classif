import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mne import io
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
import random
import copy
import pickle
from IPython.core.debugger import set_trace

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
    return pe_sub, stag_sub, idx_select


def classify_lso(pe, stag, setup, myshow=True, oversample=False, mysave=False):
    '''classify using leave_sbj_out: single sbj out, optionally oversample'''
    clf = make_pipeline(MinMaxScaler(), svm.SVC(C=1, kernel='linear',
                        probability=False))
    no_sbjs = len(stag)
    #kf = KFold(n_splits = no_sbjs)
    kf = KFold(n_splits = 2)
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
        f1_ = f1_score(pred_, y_test, average = 'micro', labels=np.unique(pred_))
        f1_each_ = f1_score(pred_, y_test, average=None, labels=[1., 2., 3.])
        acc_ = accuracy_score(pred_, y_test, normalize=True)
        predicts.append((pred_, y_test))
        scores.append(score_)
        f1s.append(f1_); accs.append(acc_); f1s_each.append(f1_each_)
        names.append(sbj_id_)
        print(test_idx)
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

def subject_h0(pe, stag, no_perm): #cv first then permutation

    #Based on Stelzer(2013). Get single subject chance accuracy estimate by permuting
    #stag labeling 'no_perm' times.
    clf = make_pipeline(svm.LinearSVC(penalty='l1',C=100., dual=False, multi_class='ovr'))
    no_sbjs = len(stag)
    kf = KFold(n_splits = no_sbjs)
    store_sbjs_h0 = []

    #shuffle full y, alternativly shuffle within fold
    #stag_shuff = [stag[i].sample(frac=1).reset_index(inplace=False, drop=True) for i in range(len(stag))]
    h0 = np.empty([no_perm, no_sbjs])
    for train_idx, test_idx in kf.split(pe, stag):
        print(test_idx)
        X = np.concatenate([pe[i].transpose() for i in train_idx], 0)
        X_test = np.concatenate([pe[i].transpose() for i in test_idx], 0)
        y = np.concatenate([stag[i]['numeric'] for i in train_idx], 0)
        y_test =  np.concatenate([stag[i]['numeric'] for i in test_idx], 0)
        for perm_idx in range(no_perm):
            clf.fit(X, y)
            pred_ = clf.predict(X_test)
            #scoring same as for real data, see upper
            np.random.shuffle(y_test) #shuffle y, rest is fixed
            f1_ = f1_score(pred_, y_test, average = 'micro', labels=np.unique(pred_))
            #acc_ = accuracy_score(pred_, y_test, normalize=True)
            h0[perm_idx,test_idx] = f1_
            print(f1_)
    return h0


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
    counter = 0
    #if '236_2' in fnames: #stag- and pe- length in '236_2' does not match TODO
    #    fnames.remove('236_2')
    for s in fnames:
        #omit if filepath is empty
        if not os.path.isfile(paths(typ=typ, sbj=s)):
            print('Folder is empty')
            counter += 1
            continue
        if typ in ['psd', 'psd_nofilt', 'psd_nofilt_ref100', 'psd_hd', 'psd_v2', 'psd_unnorm']:
            stag, pe, freqs = myload(typ=typ, sbj=s) # for psd load freqs bins

        else:
            stag, pe= myload(typ=typ, sbj=s)
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
        print(s)
        print (len(stag))
        print (pe.shape[1])
        #set_trace()
        assert len(stag) == pe.shape[-1]
        stag_list.append(stag)
        pe_list.append(pe)
        n_list.append(s)
    print('NO empty folder = {}'.format(str(counter)))
    return pe_list, stag_list, n_list, freqs

def count_stag(stag):
    no_sbj = len(stag)
    c = zip(range(no_sbj),[Counter(stag[i]['numeric']).items() for i in range(no_sbj)])
    return c

def my_rename_col(s):
    '''
    rename columns of excell sheet such that '-' => '_' (match with naming conventions
    of raw fnames)
    '''
    s_list_ = list(s.columns)
    s_new = [s_.replace('-', '_') for s_ in s_list_]
    mapper = dict(zip(s.columns, s_new))
    return s.rename(columns=mapper, copy=False)

def order_channels(raw):
    '''
    for missing channels add zeros, reorder channels  across subjects
    '''
    from config import chs_incl
    import mne
    chs = raw.info['ch_names']
    missing = [ch for ch in chs_incl if ch not in chs]
    print ('MISSING CHANNELS: %s' %missing)
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


def write_pickle(d, save_name):
    import pickle
    with open(save_name, 'wb') as f:
        pickle.dump(d, f)

def read_pickle(saved_name):
    import pickle
    with open(saved_name, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    return d


def plot_confusion_matrix(cm,  classes,
                          title,
                          normalize=False,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=0.8)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)

    plt.yticks(tick_marks, classes,fontsize=14)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Utility function to report best scores
def sc_report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))

def plot_acc_depending_scale(data):
    '''
    Plot acc for stages depedning on scales used in mspe
    '''
    import seaborn as sns
    no_rep = len(data['s1'])
    acc_dict = {'s1':[], 's2': [], 's3': [], 's4': []}
    perclass_dict =  {'s1':[], 's2': [], 's3': [], 's4': []}
    for s in data.keys():
        for r in range(no_rep):
            accav = np.asarray([data[s][r][i][0] for i in range(n_folds)]).mean(0)
            f1perclass =  np.asarray([data[s][r][i][4] for i in range(n_folds)]).mean(0)
            my_dict[s].append(accav)
            perclass_dict[s].append(f1perclass)

    perclass_dict_stat = {'s1': {}, 's2': {}, 's3': {}, 's4': {}}
    for s in perclass_dict.keys():
        perclass_dict[s] = np.array(perclass_dict[s])
    df1 = pd.DataFrame.from_dict(perclass_dict['s1'])
    df1['scale'] = ['s1'] * len(df1)
    df2 = pd.DataFrame.from_dict(perclass_dict['s2'])
    df2['scale'] = ['s2'] * len(df2)
    df3 = pd.DataFrame.from_dict(perclass_dict['s3'])
    df3['scale'] = ['s3'] * len(df3)
    df4 = pd.DataFrame.from_dict(perclass_dict['s4'])
    df4['scale'] = ['s4'] * len(df4)
    df = pd.concat([df1, df2, df3, df4])
    df.rename(columns = {0:'NREM', 1:'REM', 2:'WAKE'}, inplace=True)

    df = df.melt(value_vars= ['NREM', 'REM', 'WAKE'], id_vars='scale')
    sns.factorplot(x='scale', y='value', data=df,  kind='box', hue='variable')
    plt.show()


def remove_20hz_artif(pe, psd, stag, names, freqs, bad_sbjs):
    '''
    bad_sbjs - which subject have 20Hz artifacts (inspect plots )
    iterate over mypsd, for bad_sbjs get frequencies in 20Hz, threshold, keep bad_e_idcs and remove epochs,
    use bad_e_idcs to update pe, stag, names
    '''
    store_pe, store_psd, store_stag, store_counter= [[] for i in range(4)]
    freqs = freqs.astype(int)
    idx_freq = np.where(freqs == 20)
    for idx_sbj, sbj in enumerate(names):
        if sbj in bad_sbjs:
            print(sbj)
            #plot_data(pe[idx_sbj], psd[idx_sbj], stag[idx_sbj], names_pe[idx_sbj])
            #set_trace()
            this_data = psd[idx_sbj][idx_freq,:,:][0]
            idx_time = np.where(this_data[0,0,:] > np.percentile(this_data[0,0,:], 90))[0]
            if sbj in ['236_2'] and 0 in idx_time:
                idx_time = idx_time[1:] #see annot by HL in excel and functional.py; BL shorter
            mask_psd = np.ones(psd[idx_sbj].shape,dtype=bool)
            mask_psd[:,:,idx_time] = False
            psd_cor = psd[idx_sbj][mask_psd].reshape(psd[idx_sbj].shape[:2]+(-1,))
            mask_pe = np.ones(pe[idx_sbj].shape,dtype=bool)
            mask_pe[:,:,idx_time] = False
            pe_cor = pe[idx_sbj][mask_pe].reshape(pe[idx_sbj].shape[:2]+(-1,))
            stag_cor = stag[idx_sbj].drop(idx_time, axis=0)
            counter = len(idx_time)
            #plot_data(pe_cor, psd_cor, stag_cor, names[idx_sbj])
        else:
            pe_cor = pe[idx_sbj]
            psd_cor = psd[idx_sbj]
            stag_cor = stag[idx_sbj]
            counter = 0
        store_pe.append(pe_cor)
        store_psd.append(psd_cor)
        store_stag.append(stag_cor)
        store_counter.append(counter)
    return (store_pe, store_psd, store_stag, store_counter)


def get_unique_name_list(names1,names2):
    '''
    Take names from week2 and week5. Sort and return unique list of subjects names
    '''
    names1 = [names1[i].split('_')[0] for i in range(len(names1))]
    names2 = [names2[i].split('_')[0] for i in range(len(names2))]
    return sorted(list(set(names1 + names2)))

def find_drop_20hz(epoch):
    '''
    Drop epochs with 20Hz artifacts (egi impedance check):
        - requiers 'bad subjects' to be defined (done by visual inspection of time-freq plots)
        - drop epochs with 20Hz power higher than 90th percentile
    '''
    psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=1, fmax=30, n_fft=128, picks=slice(0,6,1))
    freqs = freqs.astype(int)
    idx_freq = np.where(freqs == 20)
    band = psds[:,:,idx_freq].squeeze()
    band = band.mean((1))
    idx_time = np.where(band[:] > np.percentile(band[:], 90))[0]
    if sbj in ['236_2'] and 0 in idx_time:
        idx_time = idx_time[1:] #see annot by HL in excel and functional.py; BL shorter
    mask_psd = np.ones(psds.shape,dtype=bool)
    mask_psd[idx_time,:,:] = False
    return(epoch.drop(idx_time))
