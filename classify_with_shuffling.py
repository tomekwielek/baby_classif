import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedShuffleSplit, KFold, RepeatedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score

from sklearn.model_selection import RandomizedSearchCV
from config import (myload, paths, report, raw_path)
from functional import sc_report
from IPython.core.debugger import set_trace
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from time import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from sklearn.preprocessing import scale

def classify_shuffle(pe, stag, idx_plot, myshow=False, check_mspe=True, null=False, n_folds=2, search=False):

    def plot_sinlge_sbj(pred, actual, f1_plot):
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(1)
        myx = range(len(pred))

        pred = [pred[i] + 0.05 for i in range(len(pred))]

        actual = [actual.tolist()[j] - 0.05 for j in range(len(actual))]

        ax.plot(pred ,  'bo', label='predicted')
        ax.scatter(myx, actual, label='actual')
        ax.set_ylim(0.8,3.2)
        ax.set_yticks([1,2,3])
        ax.yaxis.set_tick_params(width=15)
        ax.set_yticklabels(['NREM','REM','WAKE'], fontsize='large')
        ax.set_xticks(myx)
        ax.set_xlabel('Time segments [30s]')
        ax.set_xticklabels([])
        f1_str = 'Accuracy=' + str(round(f1_plot,2))
        fig.text(0.05, 0.95, f1_str, size = 22)

        ax.legend()
        plt.savefig('somefig.tif', dpi=300)
        plt.show()
        plt.close()

    no_sbjs = len(stag)
    #no_samples =  dict([(1, 45), (2, 45), (3, 45)])


    # get plot-subject
    X_plot = pe[idx_plot].T
    stag_plot = stag[idx_plot]
    y_plot = np.asarray(stag_plot)[:,1].astype('int')
    y_plot = y_plot.T

    if search == True:
        clf = RandomForestClassifier()
    else:
        clf = RandomForestClassifier(600, n_jobs=-1) #NO search
    #external cv,
    #kf = KFold(n_splits=n_folds, shuffle=True, random_state=11)
    kf = RepeatedKFold(n_splits=n_folds, n_repeats=10, random_state=11)

    #internal cv
    sskf = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=111)

    #Parameters to optimize (random search)
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 15)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
               #'bootstrap': bootstrap}
    perf = []
    for _, out_idx in kf.split(range(no_sbjs)):
        # TEST data

        X_test = [pe[i] for i in range(len(pe)) if i in out_idx]
        X_test = np.hstack(X_test).T
        y_test = [stag[i] for i in range(len(stag)) if i in out_idx]
        y_test = np.vstack(y_test)[:,1].astype('int')

        if null:
            np.random.shuffle(y_test) #shuffle y, rest is fixed

        #TRAIN and VALID data
        X_train_val = [pe[i] for i in range(len(pe)) if i not in out_idx]
        y_train_val = [stag[i] for i in range(len(stag)) if i not in out_idx]
        X_train_val = np.hstack(X_train_val)
        #get numeric labeling only
        y_train_val = np.vstack(y_train_val)[:,1].astype('int')
        X_train_val = X_train_val.T

        #scale
        #X_train_val = StandardScaler().fit_transform(X_train_val)
        #X_test = StandardScaler().fit_transform(X_test)

        #resample
        rus = RandomUnderSampler(random_state=0) #, ratio=no_samples)
        rus.fit(X_train_val, y_train_val)
        X_train_val, y_train_val = rus.sample(X_train_val, y_train_val)
        rus = RandomUnderSampler(random_state=0) #, ratio=no_samples)
        rus.fit(X_test, y_test)
        X_test, y_test = rus.sample(X_test, y_test)


        print(Counter(y_test).items())
        print(Counter(y_train_val).items())

        if search:
            # run random search
            rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                        n_iter = 100, cv=sskf, verbose=2, random_state=42, n_jobs=-1)
            start = time()
            rf_random.fit(X_train_val, y_train_val)
            print("RandomSearchCV took %.2f seconds for %d candidate parameter settings."
                    % (time() - start, len(rf_random.cv_results_['params'])))
            sc_report(rf_random.cv_results_)
            # classify TEST data
            pred_test = rf_random.predict(X_test)
            imps = []
        elif search == False:
            # NO random search
            clf.fit(X_train_val, y_train_val)
            # classify TEST data
            pred_test = clf.predict(X_test)

            # classify single plot-subject if correct fold
            if idx_plot in out_idx:
                pred_plot = clf.predict(X_plot)
                acc_pred_plot = accuracy_score(pred_plot, y_plot)

            #imps = clf.feature_importances_.reshape((5,11)) #overall importances
            #impportances = clf.feature_importances_
            imps = []
            #get class importances
            #result_imps = class_feature_importance(X_train_val, y_train_val, impportances)
            result_imps = []
            #print (json.dumps(result_imps,indent=4))

            #set_trace()
        acc = accuracy_score(pred_test, y_test)

        cmx = confusion_matrix(pred_test, y_test, labels=[1,2,3])

        recall = recall_score(pred_test, y_test, average=None)
        precission = precision_score(pred_test, y_test, average=None)
        f1_perclass = f1_score(pred_test, y_test, average=None)

        perf.append((acc, cmx, recall, precission, f1_perclass, result_imps))

    plot_sinlge_sbj(pred_plot, stag_plot['numeric'], acc_pred_plot)

    return perf


def class_feature_importance(X, Y, feature_importances):
    '''
    To get the importance according to each class, see:
     https://stackoverflow.com/questions/50201913/using-scikit-learn-to-determine-feature-importances-per-class-in-a-rf-model
    '''
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        #out[c] = dict(
        #    zip(range(M), np.mean(X[Y==c, :], axis=0)*feature_importances) )
        out[c] = np.mean(X[Y==c, :], axis=0)*feature_importances
        out[c] = out[c].reshape((5,11))
    return out
