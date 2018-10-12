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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from time import time
from imblearn.under_sampling import RandomUnderSampler

def classify_shuffle(pe, stag, myshow=False, check_mspe=True, null=False, n_folds=2, search=False):
    no_sbjs = len(stag)

    no_samples =  dict([(1, 45), (2, 45), (3, 45)])
    if search == True:
        clf = ExtraTreesClassifier()
    else:
        clf = ExtraTreesClassifier(250) #NO search

    #external cv,
    #kf = KFold(n_splits=n_folds, shuffle=True, random_state=11)
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

        elif search == False:
            # NO random search
            clf.fit(X_train_val, y_train_val)
            # classify TEST data
            pred_test = clf.predict(X_test)

            #importances = forest.feature_importances_
            #std = np.std([tree.feature_importances_ for tree in forest.estimators_],
            #     axis=0)
            #     indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            #print("Feature ranking:")

            #for f in range(X.shape[1]):
            #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        #f1_test = f1_score(pred_test, y_test, average=f1_average)
        acc = accuracy_score(pred_test, y_test)

        cm = confusion_matrix(pred_test, y_test)
        recall = recall_score(pred_test, y_test, average=None)
        precission = precision_score(pred_test, y_test, average=None)

        perf.append((acc, cm, recall, precission))

        if myshow:
            fig, axes = plt.subplots(3,1, sharex = True, figsize = [12,7])

            plot_pe(X_test, y_test, axes = axes[:2], mspe=check_mspe)
            axes[2].plot(pred_test, 'bo', label = 'prediction')
            axes[2].set_xlim(0,len(pred_test))
            axes[2].set_ylim(0.8,3.2)
            axes[2].set_yticks([1,2,3])
            axes[2].set_yticklabels(['N','R','W'], fontsize='large')
            axes[2].legend()
            times = range(len(pred_test))
            axes[2].set_xticks(times)
            axes[2].set_xticklabels('time [30s]', fontsize='large')
            f1_str = 'f1=' + str(round(f1_test,2))
            fig.text(0.05, 0.95, ['F1', f1_str], size = 22)

            report.add_figs_to_section(fig, captions='Sbj '+ name_test,
                                       section= 'MSPE')

    return perf, report
