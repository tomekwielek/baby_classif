import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedShuffleSplit, KFold, RepeatedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV
from config import (myload, paths, report, raw_path)
from functional import plot_pe, sc_report
from IPython.core.debugger import set_trace
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from time import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

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
    f1_str = 'F1=' + str(round(f1_plot,2))
    fig.text(0.05, 0.95, f1_str, size = 22)

    ax.legend()
    #plt.savefig('somefig.tif')
    plt.show()
    plt.close()

def classify_shuffle(pe, stag, idx_plot, myshow=False, check_mspe=True, null=False, n_folds=2, search=False):

    no_sbjs = len(stag)

    # get plot-subject
    X_plot = pe[idx_plot].T
    stag_plot = stag[idx_plot]
    y_plot = np.asarray(stag_plot)[:,1].astype('int')
    y_plot = y_plot.T

    no_samples =  dict([(1, 45), (2, 45), (3, 45)])
    if search == True:
        clf = ExtraTreesClassifier()
    else:
        clf = ExtraTreesClassifier(250) #NO search

    #external cv,
    kf = RepeatedKFold(n_splits=n_folds, n_repeats=5, random_state=11)

    #internal cv
    sskf = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=111)

    #Parameters to optimize (random search)
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    f1_individual_store,f1_store  = [ [] for i in range(2) ]

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

        #resample
        sampler = RandomUnderSampler(random_state=0, ratio=no_samples)
        sampler.fit(X_train_val, y_train_val)
        X_train_val, y_train_val = sampler.sample(X_train_val, y_train_val)
        sampler = RandomUnderSampler(random_state=0, ratio=no_samples)
        sampler.fit(X_test, y_test)
        X_test, y_test = sampler.sample(X_test, y_test)
        print(Counter(y_test).items())
        print(Counter(y_train_val).items())
        print('\n')

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

            # classify single plot-subject if correct fold (no sampling here)
            if idx_plot in out_idx:
                pred_plot = rf_random.predict(X_plot)

        elif search == False:
            # NO random search
            clf.fit(X_train_val, y_train_val)
            #pipe.fit(X_train_val, y_train_val)
            # classify TEST data
            pred_test = clf.predict(X_test)
            #pred_test = pipe.predict(X_test)
            # classify single plot-subject if correct fold
            if idx_plot in out_idx:
                pred_plot = clf.predict(X_plot)
                #pred_plot = pipe.predict(X_plot)
            #GET feature importance
            '''
            importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            print("Feature ranking:")

            for f in range(X_train_val.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            '''


        f1 = f1_score(pred_test, y_test, average='macro')
        f1_individual =  f1_score(pred_test, y_test, average=None)

        f1_store.append(f1)
        f1_individual_store.append(f1_individual)



    #Agregate folds
    f1_av = np.asarray(f1_store).mean()
    f1_av_individual = np.asarray(f1_individual_store).mean(0)

    f1_pred_plot = f1_score(pred_plot, y_plot, average='micro')
    if not null:
        plot_sinlge_sbj(pred_plot, stag_plot['numeric'], f1_pred_plot)


    return (f1_av, f1_av_individual)
