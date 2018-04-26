#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Huanyu ZHAI (huanyu.zhai@outlook.com)
# License: None

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import pandas as pd
from time import time

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (BaseCrossValidator, TimeSeriesSplit, 
                        GridSearchCV, KFold, learning_curve, train_test_split)
from sklearn.preprocessing import MaxAbsScaler, Normalizer, MinMaxScaler
from sklearn.utils.validation import indexable
from sklearn.utils.metaestimators import _safe_split

from .hydro_helpers.scoring import render_score
from .hydro_helpers.plotting import plot_res
from .hydro_helpers.reporting import cv_report
from .hydro_helpers._exceptions import LoadDataError

class BaseHydroModel(object):
    """ MLPModel Class for hydrological modeling (Regression).
        
        This is an MPL regressor for hydrological modeling. It optimizes the squared loss
        using quasi-Newton (LBFGS) and stochastic gradient descent (sgd or adam) methods.
        MLPModel is a child class of sklearn.neural_network.MLPRegressor, and it inherits 
        all of MLPRegressor's methodes (as well as sklearn.neural_network.BaseMultilayerPerceptron)

    """
    param_grid = {
                    'activation': ['relu', 'logistic'],
                    'hidden_layer_sizes': [(4,)],
                    'solver': ['lbfgs', 'adam', 'sgd'], 
                    'learning_rate': ['adaptive', 'constant', 'invscaling'],
                    'shuffle': [True, False],
                    'batch_size': [50, 100, 200],
                    'learning_rate_init': [0.001, 0.01, 0.0001],
                    }

    cv_scorings = {
                    'NSE': render_score('NSE'),
                    'KGE': render_score('KGE'), 
                    'RMSE': render_score('RMSE'),
                    }

    @abstractmethod
    def __init__(self):
        self._LOCAL_SCOPE_NAME = 'BaseHydroModel'
        self.X, self.y, self.X_test, self.y_test = None, None, None, None
        self.cv_split_method = KFold(n_splits=3)

    def load_data(self, path=None, loaded_data=None,
        with_prepro=False, _format='excel', with_time=False,
        colnames=["Time", "REt", "Qt"], usecols=[0,1,2],
        preprocessed=False, write_data_to_model=True):
        """ Load data for MLP model
        """
        import os.path

        message = ''
        print('Loading data...')
        _read_fun = pd.read_excel if _format=='excel' else pd.read_csv

        if preprocessed and path and os.path.exists(path):
            try:
                data = _read_fun(path)
            except Exception as e:
                message = message+' Can not load your already preprocessed data...\n'+e.message
                raise LoadDataError(message)
            else:
                self.input_data_validated = True
                self.data_ivs_done = True
                message = 'Data successfully loaded.'
                print(message, '\n', self.data.head())
                if write_data_to_model: self.data = data 
                return data
            finally:
                self.show_tips()

        if not with_time and "Time" in colnames:
            if  "Time" in colnames: colnames.remove("Time")
            if  2 in usecols: usecols.remove(2)

        if not path and not loaded_data:
            self.show_tips()
            message = 'I bet neither a path nor a loaded matrix is provided !'
            raise LoadDataError(message)
        elif path and os.path.exists(path):
            data = _read_fun(path, names=colnames, usecols=usecols)
            message = 'Data successfully loaded.'
        elif not path and loaded_data:
            data = loaded_data
            message = 'Old data successfully loaded.'
        else:
            self.show_tips()
            message = 'File does no exist, be more serious !'
            raise LoadDataError(message)

        # if with_prepro:
        #     try:
        #         self.ivs()
        #     except Exception as e:
        #         message = message+' But preprocessing failed...\n'+e.message
        #         raise LoadDataError(message)
        #     finally:
        #         self.show_tips()

        if data.shape[0] < 20:
            message = ' But only %2f samples are amazingly embarrassing to '
            'carry out a good Multilayer Perceptron Model'%self.data.shape[0]
            raise LoadDataError(message)

        self.input_data_validated = True
        if write_data_to_model: self.data = data 
        print(message, '\n', self.data.head())
        return data

    def ivs(self, shift_RE=None, shift_Q=None,
        steps_predict=None, loaded_data=None,
        write_data_to_model=True):
        """ Input variable selection
        """
        loaded_data = loaded_data or self.data

        init_length = len(loaded_data)                      # Total initial sample size
        shift_RE = shift_RE or self.shift_RE                # Shift of rainfall (related to lag)
        shift_Q = shift_Q or self.shift_Q                   # Shift of Discharge (related to lag)
        steps_predict = steps_predict or self.steps_predict # Qt+i i steps ahead to predict

        # Shifting
        df = pd.DataFrame()
        for var, shift in [('REt', shift_RE), ('Qt', shift_Q)]:
            for i in range(0, shift+1):
                colname = var if i==0 else var + '-' + str(i)
                series = loaded_data[var][max(shift_RE,shift_Q)-i:init_length-1-steps_predict-i].reset_index(drop=True)
                df[colname] = series

        colname = 'Qt+' + str(steps_predict)
        series = loaded_data.Qt[steps_predict+max(shift_RE,shift_Q):].reset_index(drop=True)
        df[colname] = series

        self.data_ivs_done = True
        if write_data_to_model:
            self.data_ivs = df
            return None
        return df

    def slice_data(self, slice_testing_set=False,
        range_for_testing=(None,None), range_for_training=(None,None),
        training_with_cv=True, write_data_to_model=True,
        loaded_data=None, percent=0.25, use_percent_split=False,
        testset_leads=True):
        """ Slice data

            Slicing the dataset into training set and testing set.
        """
        if use_percent_split:
            _fun_split = self._percent_split
        else:
            _fun_split = self._index_split
        l1,r1,l2,r2 = range_for_training[0],range_for_training[1],range_for_testing[0],range_for_testing[1]
        data = loaded_data or self.data_ivs


        if not self.input_data_validated and not loaded_data:
            raise LoadDataError("Data has not been loaded !")
        else:
            if self.data_ivs_done or loaded_data:
                pass
            else:
                warnings.warn("No input variable selection phase observed !"
                            " Currently using raw input data !")
            X,y,X_test,y_test = _fun_split(data=data, l1=l1, l2=l2, r1=r1, r2=r2,
                                        percent=percent, testset_leads=testset_leads)

        if write_data_to_model:
            self.X, self.y = X, y
            self.X_test, self.y_test = X_test, y_test

        return X, y, X_test, y_test

    def _index_split(self, data, l1, l2, r1, r2, **kwargs):
        set_train = data.iloc[l1:r1].reset_index(drop=True)
        set_test = data.iloc[l2:r2].reset_index(drop=True)
        if len(set_test)+len(set_train) > len(data):
            warnings.warn("The training set and the testing set overlaped..")
        return set_train.iloc[:,:-1],set_train.iloc[:,-1],set_test.iloc[:,:-1],set_test.iloc[:,-1]
    
    def _percent_split(self, data, percent, testset_leads, **kwargs):
        if percent > 1.0: 
            raise ValueError("Split Percentage not supposed to be greater than 1.0!")

        n = np.floor(len(data)*percent).astype('int')
        set1 = data[:n].reset_index(drop=True)
        set2 = data[n:].reset_index(drop=True)
        if not testset_leads:
            return set1.iloc[:,:-1], set1.iloc[:,-1], set2.iloc[:,:-1], set2.iloc[:,-1]
        else:
            return set2.iloc[:,:-1], set2.iloc[:,-1], set1.iloc[:,:-1], set1.iloc[:,-1]

    def scale_input(self, method='minmax', min=0.0, max=1.0,
        write_to_model=True, loaded_X_test=None, loaded_X=None):
        """ Scale Input

            Scale the input variables so that they donnot cause network paralysis (With logistic function in ANN).
        """
        X = loaded_X or self.X
        X_test = loaded_X_test or self.X_test
        
        if method == 'maxabs':
            scaler = MaxAbsScaler(copy=True)
        elif method == 'minmax':
            scaler = MinMaxScaler(copy=True, feature_range=(min, max))
        else:
            scaler = None

        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        if write_to_model:
            self.X = X_scaled
            self.X_test = X_test_scaled

        return X_scaled, X_test_scaled

    def _model_cv_setup(self, loaded_model=None, method='grid', n_splits=3,
        split_method='kfold', decisioner='NSE',
        param_grid=None, cv_scorings=None, n_jobs=1, **kwargs):
        """ Model cross-validation setup
        """
        decisioner = decisioner or self.cv_decisioner # Decisioning scorer in Hyperparameter selection (GridSearchCV)
        model = loaded_model or self._model

        cv_split_method = self.set_cv_split_method(split_method, n_splits)

        if method == 'grid':
            cv = GridSearchCV
        else:
            cv = method

        param_grid = param_grid or self.param_grid
        cv_scorings = cv_scorings or self.cv_scorings

        return cv(model, 
                  param_grid=param_grid,
                  cv=cv_split_method,
                  scoring=cv_scorings,
                  n_jobs=n_jobs, refit=decisioner)

    def model_cv(self, X=None, y=None, loaded_model=None, method='grid', n_splits=3,
        split_method='kfold', report=True, n_candidates=3, decisioner='NSE',
        param_grid=None, cv_scorings=None, n_jobs=1, save_best_model=True, **kwards):
        """ Model cross validation
        """
        cv = self._model_cv_setup(loaded_model=loaded_model, method=method,
            n_splits=n_splits, split_method=split_method, param_grid=param_grid,
            cv_scorings=cv_scorings, n_jobs=n_jobs, decisioner=decisioner)
        X = X or self.X
        y = y or self.y

        # Cross-validate
        start = time()
        cv.fit(X, y)

        print("%s took %.2f seconds for %d candidate parameter settings."
              % (cv.__class__.__name__, time()-start, len(cv.cv_results_['params'])))

        if report:
            cv_report(results=cv.cv_results_, n_top=n_candidates, score=decisioner)

        if save_best_model:
            self.deep_clone(zygote=cv.best_estimator_, overwrite=save_best_model)

        return self

    def cv(self, split_method="time_series", n_splits=3):
        """ Cross Validation

            Perform a cross validation on 
        """
        # results = []
        # # "Error_function" can be replaced by the error function of your analysis
        # for traincv, testcv in cv:
        #     probas = model.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        #     results.append( Error_function )
        pass

    def perform_learning_curve(self):
        n_jobs = 4
        train_sizes, train_scores, test_scores = learning_curve(
        self, self.X, self.y, cv=self.cv_split_method, n_jobs=n_jobs)

        print(train_sizes,'\n', train_scores,'\n', test_scores)
        return None

    def show_tips(self):
        print("===================================\n")
        print(" ML Data Driven Hydrological Model Help\n")
        print("===================================\n")
        print("Data file is ought to be present in the current working directory.\n")
        print("Data should be presented in one of the following format:\n")
        print(".csv       --- comma separated values (other delimiters also accceptable, ';', '\\t', '\\b')\n")
        print(".xls/.xlsx --- MS-Excel format. Only data in sheet No.1 are read.\n")
        print("The data file is desired to have two columns, precipitation and discharge. No gap should present.\n")

    def render_splits(self, cv_split_method=None, n_splits=None, X=None, y=None):
        method = cv_split_method or self.cv_split_method
        if (X is None and self.X is None) or (y is None and self.y is None):
            raise ValueError("No dataset provided for the split...")
        if X is None: X = self.X 
        if y is None: y = self.y

        if n_splits != None and isinstance(method,
            BaseCrossValidator.__class__):
            cv_split_method.n_splits = n_splits

        return self._gen_train_val(X, y, method)

    def _gen_train_val(self, X, y, cv_split_method):
        X, y, groups = indexable(X, y, None)
        Xs_tr, ys_tr, Xs_cv, ys_cv = [], [], [], []

        if isinstance(cv_split_method, BaseCrossValidator):
            for tr, cv in cv_split_method.split(X, y, groups):
                X_tr, y_tr = _safe_split(self, X, y, tr)
                X_cv, y_cv = _safe_split(self, X, y, cv, tr)
                Xs_tr.append(X_tr)
                Xs_cv.append(X_cv)
                ys_tr.append(y_tr)
                ys_cv.append(y_cv)
        elif cv_split_method.__name__ == 'train_test_split':
                X, X_val, y, y_val = train_test_split(
                    X, y, random_state=self._random_state,
                    test_size=self.validation_fraction)
                Xs_tr.append(X_tr)
                Xs_cv.append(X_cv)
                ys_tr.append(y_tr)
                ys_cv.append(y_cv)
        else:
            raise ValueError("Split method should be a "
                "sklearn.model_selection spliter class...")

        return Xs_tr, ys_tr, Xs_cv, ys_cv

    def set_cv_split_method(self, split_method='kfold', n_splits=3, shuffle=False):
        if split_method == 'time_series': 
            cv_split_method = TimeSeriesSplit(n_splits=n_splits)
        elif split_method == 'kfold':
            cv_split_method = KFold(n_splits=n_splits)
        else:
            cv_split_method = split_method(n_splits=n_splits, **kwargs)

        self.cv_split_method = cv_split_method
        return cv_split_method
