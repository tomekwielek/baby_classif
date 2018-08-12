import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import svm
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from config import (myload, paths, report, raw_path)
from functional import plot_pe
from scipy.stats import randint as sp_randint
from IPython.core.debugger import set_trace

average_grid = 'micro' #type of averaging of f1 scores during grid search
average_final = 'micro' #type of averaging of f1 scores for the final None or weighted

def classify_shuffle(pe, stag, myshow=False, check_mspe=True, null=False):

    pipe = Pipeline([   ('scale', MinMaxScaler()),
                        #('classify', svm.SVC(kernel='linear',probability=False,
                        ('classify', svm.SVC(probability=False,
                        class_weight='balanced'))
                        ])
    #internal cv, n splits each % test set
    sskf = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=111)
    #sskf = RepeatedKFold(n_splits=0.2,  n_repeats=10,  random_state=11)
    #external cv, 2 fold split
    no_sbjs = len(stag)
    kf = KFold(n_splits=no_sbjs, shuffle=True, random_state=11)

    perf = []
    #for out_idx in range(0, no_sbjs):
    for _, out_idx in kf.split(range(no_sbjs)):
        # TEST data
        print(out_idx)
        X_test = [pe[i] for i in range(len(pe)) if i in [out_idx][0]]
        y_test = [stag[i] for i in range(len(stag)) if i in [out_idx][0]]
        #TRAIN and VALID data
        X_train_val = [pe[i] for i in range(len(pe)) if i not in [out_idx][0]]
        y_train_val = [stag[i] for i in range(len(stag)) if i not in [out_idx][0]]

        X_train_val = np.hstack(X_train_val)

        y_train_val = np.vstack(y_train_val)[:,1].astype('int')

        X_train_val = X_train_val.T
        #TEST data
        X_test = np.hstack(X_test).T
        y_test = np.vstack(y_test)[:,1].astype('int')

        #tune the class weights using grid search to account for class imb
        weights = np.linspace(0.05, 0.95, 20) #define weights space
        fone_scorer = make_scorer(f1_score, average=average_grid) #define scorer


        '''
        gsc = GridSearchCV(
            estimator= pipe,
            param_grid={
            'classify__C': [0.1, 1, 10, 100, 1000] },
            scoring=fone_scorer,
            cv=sskf)
        '''
        '''
        param_dist = {'classify__C': sp_randint(0.1, 1000) }
        rsc = RandomizedSearchCV(
                estimator= pipe,
                param_distributions=param_dist,
                scoring=fone_scorer,
                n_iter = 40,
                cv=sskf)
        '''
        tuned_parameters = [{'classify__kernel': ['rbf', 'poly'], 'classify__gamma': [0.1, 0.01, 0.001, 0.0001],
                     'classify__C': [0.1, 1, 10, 100, 1000, 10000]}]
        gsc = GridSearchCV(
            estimator= pipe,
            param_grid= tuned_parameters,
            scoring=fone_scorer,
            cv=sskf)

        grid_result = gsc.fit(X_train_val, y_train_val)

        print ('Best C=%s' %grid_result.best_params_)
        if null:
            np.random.shuffle(y_test) #shuffle y, rest is fixed

        #get new SVM with a tuned parameters setup
        pipe = Pipeline([   ('scale', MinMaxScaler()),
                            #('classify', svm.SVC(kernel='linear', probability=False,
                            ('classify', svm.SVC(probability=False,
                            class_weight='balanced',
                            #C= list(grid_result.best_params_.values())[0] ))
                            C= list(grid_result.best_params_.values())[0],
                            gamma= list(grid_result.best_params_.values())[1],
                            kernel= list(grid_result.best_params_.values())[2]))
                            ])

        pipe.fit(X_train_val, y_train_val)
        print(grid_result.best_params_)
        # classify TEST data
        pred_test = pipe.predict(X_test)

        f1_test = f1_score(pred_test, y_test, average=average_final)
        #TODO add other metrices, includin singe ss
        perf.append(f1_test)
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

            report.add_figs_to_section(fig, captions='Sbj '+ str(out_idx),
                                       section= 'MSPE')
        print('Cum. av: {}'.format(np.asarray(perf).mean()))

    return np.asarray(perf), report


def classify_shuffle_crosstime(pe1, pe2, stag1, stag2, myshow=False, \
                    check_mspe=True, null=False, early2late=True):

    if not early2late: # if false cross gen: 5weeks - 2weeks otherwise the oposite
        pe1, pe2, stag1, stag2 = (pe2, pe1, stag2, stag1) #train t1 test t2

    pipe = Pipeline([   ('scale', MinMaxScaler()),
                        ('classify', svm.SVC(kernel='linear',probability=False,
                        class_weight='balanced'))
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
