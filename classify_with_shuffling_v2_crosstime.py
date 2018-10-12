import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import svm
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from config import (myload, paths, report, raw_path)
from functional import plot_pe
from scipy.stats import randint as sp_randint
from IPython.core.debugger import set_trace
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from time import time

def classify_shuffle_crosstime(pe1, pe2, stag1, stag2, myshow=False, \
                    check_mspe=True, null=False, early2late=True):

    if not early2late: # if false cross gen: 5weeks - 2weeks otherwise the oposite
        pe1, pe2, stag1, stag2 = (pe2, pe1, stag2, stag1) #train t1 test t2

    pipe = Pipeline([   ('scale', MinMaxScaler()),
                        ('classify', svm.SVC(kernel='linear',probability=False,
                        #class_weight='balanced'))
                        ))
                        ])

    sskf = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=111)
    #average = 'micro' #type of averaging for f1
    assert len(stag1) == len(stag2)
    no_sbjs = len(stag1)
    #loop over all subjects
    perf = []
    for out_idx in range(0, no_sbjs, 8):

        X_test = [pe2[i] for i in range(len(pe2)) if i in [out_idx]]
        y_test = [stag2[i] for i in range(len(stag2)) if i in [out_idx]]
        #training and validation on dataset '1'
        X_train_val = [pe1[i] for i in range(len(pe1)) if i not in [out_idx]]
        y_train_val = [stag1[i] for i in range(len(stag1)) if i not in [out_idx]]

        X_train_val = np.hstack(X_train_val)
        y_train_val = np.vstack(y_train_val)[:,1].astype('int')

        X_train_val = X_train_val.T

        X_test = np.hstack(X_test).T
        y_test = np.vstack(y_test)[:,1].astype('int')

        if null:
            np.random.shuffle(y_test) #shuffle y, rest is fixed


        #tune the class weights using grid search to account for class imb
        weights = np.linspace(0.05, 0.95, 20) #define weights space
        fone_scorer = make_scorer(f1_score, average=average) #define scorer

        gsc = GridSearchCV(
            estimator= pipe,
            param_grid={
            #'classify__class_weight': [{0: x, 1: 1.0-x} for x in weights]},
            'classify__C': [0.1, 1, 10, 100, 1000] },
            scoring=fone_scorer,
            cv=sskf)

        grid_result = gsc.fit(X_train_val, y_train_val)
        print ('Best C=%s' %grid_result.best_params_)
        if null:
            np.random.shuffle(y_test) #shuffle y, rest is fixed

        #get new SVM with a tuned parameters setup
        pipe = Pipeline([   ('scale', MinMaxScaler()),
                            ('classify', svm.SVC( kernel='linear',probability=False,
                            #'C' = grid_result.best_params_.values()[0],
                            class_weight='balanced',
                            C= grid_result.best_params_.values()[0]))])

        pipe.fit(X_train_val, y_train_val)
        pred_test = pipe.predict(X_test)
        f1_test = f1_score(pred_test, y_test, average=average)
        perf.append(f1_test)
        print (f1_test)
    return np.asarray(perf)
