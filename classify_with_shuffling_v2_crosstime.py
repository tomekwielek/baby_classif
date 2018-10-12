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

def classify_shuffle_crosstime(pe1, pe2, stag1, stag2, five2two, myshow=False, \
                    check_mspe=True, null=False, n_folds=2, search=False):

    no_samples =  dict([(1, 45), (2, 45), (3, 45)])
    if five2two == True:
        pe1, pe2, stag1, stag2 = (pe2, pe1, stag2, stag1)

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
    assert len(stag1) == len(stag2)
    no_sbjs = len(stag2)

    for _, out_idx in kf.split(range(no_sbjs)):
        # TEST data
        X_test = [pe2[i] for i in range(len(pe2)) if i in out_idx]
        X_test = np.hstack(X_test).T
        y_test = [stag2[i] for i in range(len(stag2)) if i in out_idx]
        y_test = np.vstack(y_test)[:,1].astype('int')

        if null:
            np.random.shuffle(y_test) #shuffle y, rest is fixed

        #TRAIN and VALID data
        X_train_val = [pe1[i] for i in range(len(pe1)) if i not in out_idx]
        y_train_val = [stag1[i] for i in range(len(stag1)) if i not in out_idx]
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
        print(Counter(y_train_val).items())
        print(Counter(y_test).items())
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
    return perf
