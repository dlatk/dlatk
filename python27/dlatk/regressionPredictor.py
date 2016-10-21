#########################################
# Regression Predictor
#
# Interfaces with FeatureWorker and scikit-learn
# to perform prediction of outcomes for lanaguage features.
#
# example: predicting satisfaction with life score given language use
#
# example usage: ./featureWorker.py --outcome_fields SWL --train_regression

import cPickle as pickle

from inspect import ismethod
import sys
import random
from itertools import combinations, izip_longest, izip
import unicodecsv as csv
import pandas as pd

from pprint import pprint
import collections
import numbers

from collections import defaultdict

#scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso, LassoCV, \
    ElasticNet, ElasticNetCV, Lars, LassoLars, LassoLarsCV, SGDRegressor, RandomizedLasso, \
    PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit, train_test_split
from sklearn.decomposition import MiniBatchSparsePCA, PCA, KernelPCA, NMF
from sklearn.lda import LDA #linear descriminant analysis
from sklearn.grid_search import GridSearchCV 
from sklearn import metrics
from sklearn.feature_selection import f_regression, SelectPercentile, SelectKBest, \
    SelectFdr, SelectFpr, SelectFwe
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin

#modified sklearns: 
from occurrenceSelection import OccurrenceThreshold
from pca_mod import RandomizedPCA #allows percentage input

#statsmodels (for weighted least squares)
#import scikits.statsmodels.api as sm

###spams
#sys.path.append('./spams')
#import spams

#scipy
import pylab as pl
from scipy.stats import zscore, ttest_rel, ttest_1samp
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import t
from scipy.sparse import csr_matrix
import numpy as np
from numpy import sqrt, array, std, mean, ceil, absolute, append, log
from numpy.linalg.linalg import LinAlgError


import math

DEFAULT_MAX_PREDICT_AT_A_TIME = 100000

#infrastructure
from classifyPredictor import ClassifyPredictor
from mysqlMethods import mysqlMethods as mm

#alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
def alignDictsAsXy(X, y, sparse = False, returnKeyList = False, keys = None):
    """turns a list of dicts for x and a dict for y into a matrix X and vector y"""
    
    if not keys: 
        keys = frozenset(y.keys())
        if sparse:
            keys = keys.intersection([item for sublist in X for item in sublist]) #union X keys
        else:
            keys = keys.intersection(*[x.keys() for x in X]) #intersect X keys
        keys = list(keys) #to make sure it stays in order
    listy = None
    try:
        listy = map(lambda k: float(y[k]), keys)
    except KeyError:
        print "!!!WARNING: alignDictsAsXy: gid not found in y; groups are being dropped (bug elsewhere likely)!!!!"
        ykeys = y.keys()
        keys = [k for k in keys if k in ykeys]
        listy = map(lambda k: float(y[k]), keys)
        
    if sparse:
        keyToIndex = dict([(keys[i], i) for i in xrange(len(keys))])
        row = []
        col = []
        data = []
        for c in xrange(len(X)):
            column = X[c]
            for keyid, value in column.iteritems():
                if keyid in keyToIndex:
                    row.append(keyToIndex[keyid])
                    col.append(c)
                    data.append(value)

        assert all([isinstance(x,numbers.Number) for x in data]), "Data is corrupt, there are non float elements in the group norms (some might be NULL?)"
        sparseX = csr_matrix((data,(row,col)), shape = (len(keys), len(X)), dtype=np.float)
        if returnKeyList:
            return (sparseX, array(listy), keys)
        else:
            return (sparseX, array(listy))
        
    else: 
        listX = map(lambda k: [x[k] for x in X], keys)
        if returnKeyList:
            return (array(listX), array(listy), keys)
        else:
            return (array(listX), array(listy))

def alignDictsAsXywithFeats2(X, y, sparse = False, returnKeyList = False, keys = None, feats = None):
    """turns a list of dicts for x and a dict for y into a matrix X and vector y"""
    
    if not keys: 
        keys = frozenset(y.keys())
        if sparse:
            keys = keys.intersection([item for sublist in X for item in sublist]) #union X keys
        else:
            keys = keys.intersection(*[x.keys() for x in X]) #intersect X keys
        keys = list(keys) #to make sure it stays in order
    listy = None
    try:
        listy = map(lambda k: float(y[k]), keys)
    except KeyError:
        print "!!!WARNING: alignDictsAsXy: gid not found in y; groups are being dropped (bug elsewhere likely)!!!!"
        ykeys = y.keys()
        keys = [k for k in keys if k in ykeys]
        listy = map(lambda k: float(y[k]), keys)
        
    # keys = list of group_ids
    if sparse:
        keyToIndex = dict([(keys[i], i) for i in xrange(len(keys))])
        row = []
        col = []
        data = []
        # X is [{group_id: group_norm for feat1, group_id : gns feat1 ...} ,... ]
        featsToColIndices = dict()
        # X has nbFeats as len, but not each dict in X has same length
        for c in xrange(len(X)):
            feat = feats[c]
            column = X[c]
            for keyid, value in column.iteritems():
                if keyid in keyToIndex:
                    row.append(keyToIndex[keyid]) # appending the number of the row that group_id was for this gn
                    col.append(c)
                    data.append(value)
        sparseX = csr_matrix((data,(row,col)), shape = (len(keys), len(X)))
        if returnKeyList:
            return (sparseX, array(listy), keys)
        else:
            return (sparseX, array(listy))
        
    else: 
        listX = map(lambda k: [x[k] for x in X], keys)
        if returnKeyList:
            return (array(listX), array(listy), keys)
        else:
            return (array(listX), array(listy))

def alignDictsAsXyWithFeats(X, y, feats):
    """turns a list of dicts for x and a dict for y into a matrix X and vector y"""
    keys = frozenset(y.keys())
    # keys = keys.intersection(*[x.keys() for x in X])
    keys = keys.intersection(X.keys())  
    keys = list(keys) #to make sure it stays in order
    feats = list(feats)
    listy = map(lambda k: y[k], keys)
    listX = map(lambda k: map(lambda l: X[k][l], feats), keys)
    return (listX, listy)

def alignDictsAsy(y, *yhats, **kwargs):
    keys = None
    if not keys in kwargs: 
        keys = frozenset(y.keys())
        for yhat in yhats:
            keys = keys.intersection(yhat.keys())
    else:
        keys = kwargs['keys']
    listy = map(lambda k: y[k], keys)
    listyhats = []
    for yhat in yhats: 
        listyhats.append(map(lambda k: yhat[k], keys))
    return tuple([listy]) + tuple(listyhats)

class RegressionPredictor:
    """Handles prediction of continuous outcomes"""

    #cross validation parameters:
    # Model Parameters 
    cvParams = {
        'linear': [{'fit_intercept':[True]}],
        'ridge': [
            {'alpha': [100]}, 
        ],
        'ridge10000': [
            {'alpha': [10000]}, #topics age
        ],
        'ridge1000': [
            {'alpha': [1000]}, #rpca counties, overfit?
        ],
        'ridge250': [
            {'alpha': [250]}, #PCA counties
        ],
        'ridge100000': [
            {'alpha': [100000]}, 
        ],
        'ridge100': [
            {'alpha': [100]}, 
        ],
        'ridgecv': [
            #{'alphas': np.array([100000, 500000, 250000, 25000, 10000, 2500, 1000, 100, 10])}, 
            #{'alphas': np.array([100000, 500000, 250000, 25000, 10000])},
            #{'alphas': np.array([250000, 100000, 1000000, 2500000, 10000000, 25000000, 100000000])}, #personality, n-grams + 2000 topics
            {'alphas': np.array([1, .01, .0001, 100, 10000, 1000000])}, #first-pass
            #{'alphas': np.array([1000, 1, .1, 10, 100, 10000, 100000])}, #user-level low num users (~5k) or counties
            #{'alphas': np.array([1000, 10000, 100000, 1000000])}, #user-level low num users (~5k) or counties
            #{'alphas': np.array([1000, 100, 10000, 10, 1])}, #county achd (need low for controls)
            #{'alphas': np.array([10000, 100000, 1000, 1000000, 100])}, #message quality
            #{'alphas': np.array([1, .1, .01, .001, .0001, .00001, .000001])} 
            #{'alphas': np.array([.01, .001, .1, 1, 10, .0001])} #message-level rel_freq no std
            #{'alphas': np.array([.001])},
            #{'alphas': np.array([100, 1000, 10])},
            #{'alphas': np.array([10, .1, .25, 1, 2.5, 100])}, #user-level low num users (~5k) or counties
            #{'alphas': np.array([1000, 1, 10000, 100000, 1000000])}, #user-level medium num users (~25k)
            #{'alphas': np.array([1000, 100, 10000, 10, 100000, 1])}, #message-level sparse binary
            #{'alphas': np.array([100000, 25000, 250000, 1000000, 10000, 2500, 1000, 250, 1])}, #user-level swl (sparse)
            #{'alphas': np.array([10000, 1000, 100000])}, #age
            #{'alphas': np.array([2500, 10000, 1000, 250, 25000])}, #message-level perma, personality pca with(out?) whiten
            #{'alphas': np.array([33000, 10000, 100000, 330000, 1000000, 3300000])}, #2k features << 10k examples
            #{'alphas': np.array([25000, 100000, 1000000, 10000000, 10000]), 'fit_intercept': False}, #1to3 grams and personality (not tsted)
            #{'alphas': np.array([25000, 100000, 250000, 1000000, 2500000, 10000000, 100000000 ])}, #county-level pca ngrams health
            #{'alphas': None}
            #{'alpha': [.001, .01, .0001]}
            ],

        'ridgefirstpasscv': [
            {'alphas': np.array([1, .01, .0001, 100, 10000, 1000000])}, 
            ],
        'ridgehighcv': [
            {'alphas': np.array([10,100, 1, 1000, 10000, 100000, 1000000])}, 
        ],
        'ridgelowcv': [
            {'alphas': np.array([.01, .1, .001, 1, .0001, .00001])}, 
        ],
        'rpcridgecv': [
            #{'alphas': np.array([100, 1000, 10000]), 'component_percs':np.array([.01, .02154, .0464, .1, .2154, .464, 1])}, #standard
            #{'alphas': np.array([10, 100, 1000]), 'component_percs':np.array([.01, .02154, .0464, .1, .2154, .464, 1])}, #standard
            #{'alphas': np.array([10, 100, 1000]), 'component_percs':np.array([0.1])}, #standard
            {'alphas': np.array([.00001]), 'component_percs':np.array([.01, .02154, .0464, .1, .2154, .464, 1])}, #one penalization
            ],
        'lasso': [
            #{'alpha': [10000, 25000, 1000, 2500, 100, 250, 1, 25, .01, 2.5, .01, .25, .001, .025, .0001, .0025, .00001, .00025, 100000, 250000, 1000000]}, 
            #{'alpha': [0.1, 0.01, 0.001, 0.0001], 'max_iter':[1500]}, 
            #{'alpha': [10000, 100, 1, 0.1, 0.01, 0.001, 0.0001]}, 
            {'alpha': [0.001], 'max_iter':[1500]}, #1to3gram, personality (best => 0.01, fi=true: .158)
            ],
        'lassocv': [
            #{'alpha': [10000, 25000, 1000, 2500, 100, 250, 1, 25, .01, 2.5, .01, .25, .001, .025, .0001, .0025, .00001, .00025, 100000, 250000, 1000000]}, 
            #{'n_alphas': [24], 'max_iter':[2200]}, #final county-level prediction
            #{'n_alphas': [15], 'max_iter':[1500]}, 
            #{'n_alphas': [12], 'max_iter':[1300]}, #message-level perma/swl
            {'n_alphas': [12], 'max_iter':[2200]},
            #{'n_alphas': [9], 'max_iter':[1100]}, #min-decent results
            #{'n_alphas': [4], 'max_iter':[800]}, #quick test
            ],
        'elasticnet': [
            #{'alpha': [10000, 25000, 1000, 2500, 100, 250, 1, 25, .01, 2.5, .01, .25, .001, .025, .0001, .0025, .00001, .00025, 100000, 250000, 1000000]}, 
            #{'alpha': [10000, 100, 1, 0.1, 0.01, 0.001, 0.0001], 'max_iter' : [2000]}, 
            #{'alpha': [0.01], 'max_iter' : [1500]}, 
            #{'alpha': [0.1, 0.01, 0.001], 'max_iter' : [1500]}, 
            {'alpha': [0.001], 'max_iter' : [1500], 'rho': [0.8]}, 
            ],
        'elasticnetcv': [
            #{'alpha': [10000, 25000, 1000, 2500, 100, 250, 1, 25, .01, 2.5, .01, .25, .001, .025, .0001, .0025, .00001, .00025, 100000, 250000, 1000000]}, 
            #{'n_alphas': [10], 'max_iter':[1600], 'rho' : [.1, .5, .7, .9, .95, .99, 1]}, 
            #{'n_alphas': [11], 'max_iter':[1300], 'l1_ratio' : [.95], 'n_jobs': [10]}, 
            {'n_alphas': [11], 'max_iter':[1300], 'l1_ratio' : [.01, .05, .1, .5, .9, .95, .99], 'n_jobs': [8]}, 
            #{'n_alphas': [9], 'max_iter':[1100], 'l1_ratio' : [.95], 'n_jobs': [6]}, 
            #{'n_alphas': [7], 'max_iter':[1100], 'l1_ratio' : [.95], 'n_jobs': [6]}, 
            ],
        'lars': [
            {}, 
            ],
        'lassolars': [
            #{'alpha': [10000, 100, 1, 0.1, 0.01, 0.001, 0.0001]}, 
            {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}, 
            ],
        'lassolarscv': [
            {'max_n_alphas': [60], 'max_iter':[1000]}, 
            ],

        'svr': [
            #{'kernel':['linear'], 'C':[.01, .25, .001, .025, .0001, .0025, .00001], 'epsilon': [1, 0.1, 0.01, 0.001]}#swl
            {'kernel':['linear'], 'C':[.01, .001, .0001, .00001, .000001, .0000001], 'epsilon': [0.25]}#personality
            #{'kernel':['rbf'], 'gamma': [0.1, 1, 0.001, .0001], 'C':[.1, .01, 1, .001, 10]}#swl
            ],
        'sgdregressor': [
            {'alpha':[250000, 25000, 250, 25, 1, 0.1, .01, .001, .0001, .00001, .000001], 'penalty':['l1']}#testing for personality
            ],
        'extratrees':[
            #{'n_estimators': [20], 'n_jobs': [10], 'random_state': [42], 'compute_importances' : [True]},
            {'n_estimators': [1000], 'n_jobs': [12], 'random_state': [42]},
            ],
        'par':[
            #{'C': [.01], 'random_state': [42], 'verbose': [1], 'shuffle': [False], 'epsilon': [0.01], 'n_iter': [10]},
            {'C': [.01, .1, .001], 'random_state': [42], 'verbose': [1], 'shuffle': [False], 'epsilon': [0.01, .1, 1], 'n_iter': [10]},
            ],
        
       }
    modelToClassName = {
        'lasso' : 'Lasso',
        'lassocv' : 'LassoCV',
        'elasticnet' : 'ElasticNet',
        'elasticnetcv' : 'ElasticNetCV',
        'lassolars' : 'LassoLars',
        'lassolarscv' : 'LassoLarsCV',
        'lars' : 'Lars',
        'ridge' : 'Ridge',
        'ridge250' : 'Ridge',
        'ridge10000' : 'Ridge',
        'ridge100000' : 'Ridge',
        'ridge1000' : 'Ridge',
        'ridge100' : 'Ridge',
        'ridgecv' : 'RidgeCV',
        'ridgefirstpasscv' : 'RidgeCV',
        'ridgehighcv' : 'RidgeCV',
        'ridgelowcv' : 'RidgeCV',
        'rpcridgecv' : 'RPCRidgeCV',
        'linear' : 'LinearRegression',
        'svr': 'SVR',
        'sgdregressor': 'SGDRegressor',
        'extratrees': 'ExtraTreesRegressor',
        'par': 'PassiveAggressiveRegressor',
        }
    
    modelToCoeffsName = {
        'lasso' : 'coef_',
        'lassocv' : 'coef_',
        'elasticnet' : 'coef_',
        'elasticnetcv' : 'coef_',
        'lassolars' : 'coef_',
        'lassolarscv' : 'coef_',
        'lars' : 'coef_',
        'ridge' : 'coef_',
        'ridge250' : 'coef_',
        'ridge10000' : 'coef_',
        'ridge100000' : 'coef_',
        'ridge1000' : 'coef_',
        'ridge100' : 'coef_',
        'ridgecv' : 'coef_',
        'ridgefirstpasscv' : 'coef_',
        'ridgehighcv' : 'coef_',
        'ridgelowcv' : 'coef_',
        'rpcridgecv' : 'coef_',
        'linear' : 'coef_',
        'svr': 'coef_',
        'sgdregressor': 'coef_',
        'extratrees': 'feature_importances_',
        'par': 'coef_',
        'randomizedlasso': 'scores_'
        }
    #cvJobs = 3 #when lots of data 
    #cvJobs = 6 #normal
    cvJobs = 8 #resource-heavy
    cvFolds = 3
    chunkPredictions = False #whether or not to predict in chunks (good for keeping track when there are a lot of predictions to do)
    maxPredictAtTime = 60000
    backOffPerc = .05 #when the num_featrue / training_insts is less than this backoff to backoffmodel
    #backOffModel = 'ridgecv'
    backOffModel = 'linear'

    # feature selection:
    featureSelectionString = None
    #featureSelectionString= 'ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute="auto", max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, rho=None)'
    #featureSelectionString = 'RandomizedLasso(random_state=self.randomState, n_jobs=self.cvJobs, normalize=False)'
    #featureSelectionString = 'ExtraTreesRegressor(n_jobs=self.cvJobs, random_state=42, compute_importances=True)'
    #featureSelectionString = 'SelectKBest(f_regression, k=int(len(y)/3))'
    #featureSelectionString = 'SelectFpr(f_regression)' #this is correlation feature selection
    #featureSelectionString = 'SelectFwe(f_regression, alpha=60.0)' #this is correlation feature selection w/ correction
    #featureSelectionString = 'SelectFwe(f_regression, alpha=0.1)' #this is correlation feature selection w/ correction
    #featureSelectionString = 'SelectPercentile(f_regression, 33)'#1/3 of features
    #featureSelectionString = 'Lasso(alpha=0.15)'
    #featureSelectionString = 'OccurrenceThreshold(threshold=1.5)'
    #featureSelectionString = 'Pipeline([("univariate_select", SelectPercentile(f_regression, 33)), ("L1_select", RandomizedLasso(random_state=42, n_jobs=self.cvJobs))])'
    #featureSelectionString = 'Pipeline([("1_univariate_select", SelectFwe(f_regression, alpha=60.0)), ("2_rpca", RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/max(1.5,len(self.featureGetters)))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])'
    

    #featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=60.0)), ("3_rpca", RandomizedPCA(n_components=max(int(X.shape[0]/max(1.5,len(self.featureGetters))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])'
    #featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=60.0)), ("3_rpca", RandomizedPCA(n_components=(X.shape[0]/4), random_state=42, whiten=False, iterated_power=3))])'
    #featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=100.0)), ("3_rpca", RandomizedPCA(n_components=max(int(X.shape[0]/(5.0*len(self.featureGetters))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])'
    #featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=70.0)), ("3_rpca", RandomizedPCA(n_components=.4/len(self.featureGetters), random_state=42, whiten=False, iterated_power=3, max_components=X.shape[0]/max(1.5, len(self.featureGetters))))])'
    #featureSelectionString = 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=(X.shape[0]/100.0))), ("2_univariate_select", SelectFwe(f_regression, alpha=70.0)), ("3_rpca", RandomizedPCA(n_components=.4, random_state=42, whiten=False, iterated_power=3, max_components=X.shape[0]/max(1.5, len(self.featureGetters))))])'

    # dimensionality reduction (TODO: make a separate step than feature selection)
    #featureSelectionString = 'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/max(1.5,len(self.featureGetters)))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)' #smaller among 10% or number of rows / number of feature tables
    #featureSelectionString = 'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/2)), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)'#smaller among 10% or number of rows / 2
    #featureSelectionString = 'RandomizedPCA(n_components=min(X.shape[1], int(X.shape[0]/4)), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=min(X.shape[1], int(X.shape[1]/10)), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=min(X.shape[1], 2000), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=min(int(X.shape[0]*1.5), X.shape[1]), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'PCA(n_components=min(int(X.shape[1]*.10), X.shape[0]), whiten=False)'
    #featureSelectionString = 'PCA(n_components=0.99, whiten=False)'
    #featureSelectionString = 'PCA(n_components=0.95, whiten=False)'
    #featureSelectionString = 'VERPCA(n_components=0.999, whiten=False, max_components_ratio = min(1, X.shape[0]/float(X.shape[1])))'
    #featureSelectionString = 'KernelPCA(n_components=int(X.shape[1]*.02), kernel="rbf", degree=3, eigen_solver="auto")'  
    #featureSelectionString = \
    #    'MiniBatchSparsePCA(n_components=int(X.shape[1]*.05), random_state=42, alpha=0.01, chunk_size=20, n_jobs=self.cvJobs)'

    featureSelectMin = 30 #must have at least this many features to perform feature selection
    featureSelectPerc = 1.00 #only perform feature selection on a sample of training (set to 1 to perform on all)
    #featureSelectPerc = 0.20 #only perform feature selection on a sample of training (set to 1 to perform on all)

    testPerc = .20 #percentage of sample to use as test set (the rest is training)
    randomState = 42 #percentage of sample to use as test set (the rest is training)
    #randomState = 64 #percentage of sample to use as test set (the rest is training)

    trainingSize = 1000000 #if this is smaller than the training set, then it will be reduced to this. 

    def __init__(self, og, fgs, modelName = 'ridge'):
        #initialize regression predictor
        self.outcomeGetter = og

        #setup feature getters:

        if not isinstance(fgs, collections.Iterable):
            fgs = [fgs]
        self.featureGetters = fgs
        self.featureGetter = fgs[0] #legacy support

        #setup other params / instance vars
        self.modelName = modelName
        self.regressionModels = dict()
        self.scalers = dict()
        self.fSelectors = dict()
        self.featureNames = [] #holds the order the features are expected in
        self.multiFSelectors = None
        self.multiScalers = None
        self.multiXOn = False #whether multiX was used for training

    def train(self, standardize = True, sparse = False, restrictToGroups = None, groupsWhere = ''):
        """Train Regressors"""

        ################
        #1. setup groups
        (groups, allOutcomes, allControls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        if restrictToGroups: #restrict to groups
            rGroups = restrictToGroups
            if isinstance(restrictToGroups, dict):
                rGroups = [item for sublist in restrictToGroups.values() for item in sublist]
            groups = groups.intersection(rGroups)
            for outcomeName, outcomes in allOutcomes.iteritems():
                allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
            for controlName, controlValues in allControls.iteritems():
                allControls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % (len(groups))

        ####################
        #2. get data for Xs:
        (groupNormsList, featureNamesList) = ([], [])
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        for fg in self.featureGetters:
            (groupNorms, featureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            groupNormValues = [groupNorms[feat] for feat in featureNames] #list of dictionaries of group_id => group_norm
            groupNormsList.append(groupNormValues)
            # print featureNames[:10]#debug
            featureNamesList.append(featureNames)
            if not XGroups:
                XGroups = set(getGroupsFromGroupNormValues(groupNormValues))
            else:
                XGroups = XGroups & getGroupsFromGroupNormValues(groupNormValues) #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should
        #XGroups = XGroups & groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        XGroups = XGroups.union(groups) #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        
        ################################
        #2b) setup control data:
        controlValues = allControls.values()
        self.controlsOrder = allControls.keys()

        if controlValues:
            groupNormsList.append(controlValues)

        #########################################
        #3. train for all possible ys:
        self.multiXOn = True
        (self.regressionModels, self.multiScalers, self.multiFSelectors) = (dict(), dict(), dict())
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            multiXtrain = list()
            trainGroupsOrder = list(XGroups & set(outcomes.keys()))
            for i in xrange(len(groupNormsList)):
                groupNormValues = groupNormsList[i]
                #featureNames = featureNameList[i] #(if needed later, make sure to add controls to this)
                (Xdicts, ydict) = (groupNormValues, outcomes)
                print "  (feature group: %d)" % (i)
                # trainGroupsOrder is the order of the group_norms
                (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                # (Xtrain, ytrain) = alignDictsAsXywithFeats2(Xdicts, ydict, sparse=True, keys = trainGroupsOrder, feats = featureNamesList[i])
                # MAARTEN
                if len(ytrain) > self.trainingSize:
                    Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                multiXtrain.append(Xtrain)
                print "   [Train size: %d ]" % (len(ytrain))

            #############
            #4) fit model
            (self.regressionModels[outcomeName], self.multiScalers[outcomeName], self.multiFSelectors[outcomeName]) = \
                self._multiXtrain(multiXtrain, ytrain, standardize, sparse = sparse)

        print "\n[TRAINING COMPLETE]\n"
        self.featureNamesList = featureNamesList

    ##################
    ## Old testing Method (random split rather than cross-val)
    def test(self, standardize = True, sparse = False, saveModels = False, blacklist = None, groupsWhere = ''):
        """Tests classifier, by pulling out random testPerc percentage as a test set"""
        
        print
        print "USING BLACKLIST: %s" %str(blacklist)
        #1. get data possible ys (outcomes)
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        print "[number of groups: %d]" % len(groups)

        #2. get data for X:
        (groupNorms, featureNames) = (None, None)
        if len(self.featureGetters) > 1: 
            print "\n!!! WARNING: multiple feature tables passed in rp.test only handles one for now.\n        try --combo_test_regression or --predict_regression !!!\n"
        if sparse:
            if blacklist:
                print "\n!!!WARNING: USING BLACKLIST WITH SPARSE IS NOT CURRENTLY SUPPORTED!!!\n"
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsSparseFeatsFirst(groups)
        else:
            if blacklist: print "USING BLACKLIST: %s" %str(blacklist)
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups, blacklist = blacklist)
        groupNormValues = groupNorms.values() #list of dictionaries of group => group_norm
        controlValues = controls.values() #list of dictionaries of group=>group_norm
    #     this will return a dictionary of dictionaries

        #3. test classifiers for each possible y:
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            (X, y) = alignDictsAsXy(groupNormValues+controlValues, outcomes, sparse)
            print "[Initial size: %d]" % len(y)
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=self.testPerc, random_state=self.randomState)
            if len(ytrain) > self.trainingSize:
                Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
            print "[Train size: %d    Test size: %d]" % (len(ytrain), len(ytest))

            (regressor, scaler, fSelector) = self._train(Xtrain, ytrain, standardize)
            ypred = self._predict(regressor, Xtest, scaler = scaler, fSelector = fSelector)
            R2 = metrics.r2_score(ytest, ypred)
            #R2 = r2simple(ytest, ypred)
            print "*R^2 (coefficient of determination): %.4f"% R2
            print "*R (sqrt R^2):                       %.4f"% sqrt(R2)
            #ZR2 = metrics.r2_score(zscore(list(ytest)), zscore(list(ypred)))
            #print "*Standardized R^2 (coef. of deter.): %.4f"% ZR2
            #print "*Standardized R (sqrt R^2):          %.4f"% sqrt(ZR2)
            print "*Pearson r:                          %.4f (p = %.5f)"% pearsonr(ytest, ypred)
            print "*Spearman rho:                       %.4f (p = %.5f)"% spearmanr(ytest, ypred)
            mse = metrics.mean_squared_error(ytest, ypred)
            print "*Mean Squared Error:                 %.4f"% mse
            #print "*Root Mean Squared Error:            %.5f"% sqrt(float(mse))
            if saveModels: 
                (self.regressionModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName]) = (regressor, scaler, fSelector)

        print "\n[TEST COMPLETE]\n"

    #####################################################
    ####### Meain Testing Method ########################
    def testControlCombos(self, standardize = True, sparse = False, saveModels = False, blacklist = None, noLang = False, 
                          allControlsOnly = False, comboSizes = None, nFolds = 2, savePredictions = False, weightedEvalOutcome = None, residualizedControls = False, groupsWhere = ''):
        """Tests regressors, by cross-validating over folds with different combinations of controls"""
        
        ###################################
        #1. setup groups for random folds
        if blacklist: print "USING BLACKLIST: %s" %str(blacklist)
        (groups, allOutcomes, allControls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        print "[number of groups: %d (%d Folds)]" % (len(groups), nFolds)
        random.seed(self.randomState)
        groupList = sorted(list(groups), reverse=True)
        random.shuffle(groupList)
        groupFolds =  [x for x in foldN(groupList, nFolds)]

        ####
        #1a: setup weightedEval
        wOutcome = None
        if weightedEvalOutcome:
            try: 
                wOutcome = allOutcomes[weightedEvalOutcome]
                del allOutcomes[weightedEvalOutcome]
            except KeyError:
                print "Must specify weighted eval outcome as a regular outcome in order to get data."
                sys.exit(1)

        ####################
        #2. get data for Xs:
        (groupNormsList, featureNamesList) = ([], [])
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        for fg in self.featureGetters:
            #(groupNorms, featureNames) = (None, None)
            if blacklist:
                print "!!!WARNING: USING BLACKLIST WITH SPARSE IS NOT CURRENTLY SUPPORTED!!!"
            (groupNorms, featureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            groupNormValues = groupNorms.values() #list of dictionaries of group_id => group_norm
            groupNormsList.append(groupNormValues)
            featureNamesList.append(featureNames)
            if not XGroups:
                XGroups = set(getGroupsFromGroupNormValues(groupNormValues))
            else:
                XGroups = XGroups & getGroupsFromGroupNormValues(groupNormValues) #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should
        XGroups = XGroups & groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        
        ################################
        #2b) setup control combinations:
        controlKeys = allControls.keys()
        scores = dict() #outcome => control_tuple => [0],[1] => scores= {R2, R, r, r-p, rho, rho-p, MSE, train_size, test_size, num_features,yhats}
        if not comboSizes:
            comboSizes = xrange(len(controlKeys)+1)
            if allControlsOnly:
                comboSizes = [0, len(controlKeys)]
        for r in comboSizes:
            for controlKeyCombo in combinations(controlKeys, r):
                controls = dict()
                if len(controlKeyCombo) > 0:
                    controls = dict([(k, allControls[k]) for k in controlKeyCombo])
                controlKeyCombo = tuple(controlKeyCombo)
                print "\n\n|COMBO: %s|" % str(controlKeyCombo)
                print '='*(len(str(controlKeyCombo))+9)
                controlValues = controls.values() #list of dictionaries of group=>group_norm
                thisGroupNormsList = list(groupNormsList)
                if controlValues:
                    thisGroupNormsList.append(controlValues)

                #########################################
                #3. test classifiers for each possible y:
                for outcomeName, outcomes in sorted(allOutcomes.items()):
                    thisOutcomeGroups = set(outcomes.keys()) & XGroups
                    if not outcomeName in scores:
                        scores[outcomeName] = dict()
                    for withLanguage in xrange(2):
                        if withLanguage: 
                            if noLang or (allControlsOnly and (r > 1) and (r < len(controlKeys))):#skip to next
                                continue
                            print "\n= %s (w/ lang.)=\n%s"%(outcomeName, '-'*(len(outcomeName)+14))
                        elif controlValues: 
                            print "\n= %s (NO lang.)=\n%s"%(outcomeName, '-'*(len(outcomeName)+14))
                        else: #no controls in this iteration
                            continue
                        testStats = {'R2_folds': [], 'r_folds': [], 'r_p_folds': [], 'mse_folds': [], 'mae_folds': [], 'train_mean_mae_folds': []}

                        if wOutcome:
                            testStats.update({'rwghtd_folds' : [], 'rwghtd_p_folds' : []})
                        predictions = {}

                        #################################3
                        ## 2 f) calculate residuals, if applicable:
                        nonresOutcomes = dict(outcomes) #backup for accuracy calc. #new outcomes become residuals
                        resControlPreds = {} #predictions form controls only
                        if residualizedControls and controlValues and withLanguage:
                            #TODO: make this and below a function:
                            print "CREATING RESIDUALS"
                            #creating residuals:
                            for testChunk in xrange(0, len(groupFolds)):
                                print " Residual fold %d " % (testChunk)
                                trainGroups = set()
                                for chunk in (groupFolds[:testChunk]+groupFolds[(testChunk+1):]):
                                    for c in chunk:
                                        trainGroups.add(c)
                                testGroups = set(groupFolds[testChunk])
                                #set static group order across features:
                                trainGroupsOrder = list(thisOutcomeGroups & trainGroups)
                                testGroupsOrder = list(thisOutcomeGroups & testGroups)
                                testSize = len(testGroupsOrder)

                                groupNormValues = thisGroupNormsList[-1]
                                (Xdicts, ydict) = (groupNormValues, outcomes)
                                print "  [Initial size: %d]" % (len(ydict))
                                (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                                (Xtest, ytest) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = testGroupsOrder)

                                assert len(ytest) == testSize, "ytest not the right size"
                                if len(ytrain) > self.trainingSize:
                                    Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                                (res_regressor, res_multiScalers, res_multiFSelectors) = self._multiXtrain([Xtrain], ytrain, standardize, sparse = sparse)
                                res_ypred = self._multiXpredict(res_regressor, [Xtest], multiScalers = res_multiScalers, multiFSelectors = res_multiFSelectors, sparse = sparse)
                                #DEBUG: random sort
                                #random.shuffle(res_ypred)
                                resControlPreds.update(dict(izip(testGroupsOrder,res_ypred)))

                                R2 = metrics.r2_score(ytest, res_ypred)
                                mse = metrics.mean_squared_error(ytest, res_ypred)
                                mae = metrics.mean_absolute_error(ytest, res_ypred)
                                train_mean = mean(ytrain)
                                train_mean_mae = metrics.mean_absolute_error(ytest, [train_mean]*len(ytest))
                                print "  residual fold R^2: %.4f (MSE: %.4f; MAE: %.4f; mean train mae: %.4f)"% (R2, mse, mae, train_mean_mae)

                            #update outcomes to be residuals:
                            for gid, value in outcomes.items():
                                try: 
                                    #print "true outcome: %.4f, controlPred: %.4f" % (value, resControlPreds[gid])#debug
                                    outcomes[gid] = value - resControlPreds[gid]
                                    #print " new value %.4f" % outcomes[gid] #debug
                                except KeyError:
                                    print " warn: no control prediction found for gid %s, removing from outcomes" % str(gid)
                                    del outcomes[gid]
                            print "DONE CREATING RESIDUALS for %s %s\n" % (outcomeName, str(controlKeyCombo))
                        
                        ###############################
                        #3a) iterate over nfold groups:
                      
                        for testChunk in xrange(0, len(groupFolds)):
                            trainGroups = set()
                            for chunk in (groupFolds[:testChunk]+groupFolds[(testChunk+1):]):
                                for c in chunk:
                                    trainGroups.add(c)
                            testGroups = set(groupFolds[testChunk])
                            #set static group order across features:
                            trainGroupsOrder = list(thisOutcomeGroups & trainGroups)
                            testGroupsOrder = list(thisOutcomeGroups & testGroups)
                            testSize = len(testGroupsOrder)
                            print "Fold %d " % (testChunk)

                            ###########################################################################
                            #3b)setup train and test data (different X for each set of groupNormValues)
                            (multiXtrain, multiXtest, ytrain, ytest) = ([], [], None, None) #ytrain, ytest should be same across tables
                            num_feats = 0;
                            #get the group order across all
                            gnListIndices = range(len(thisGroupNormsList))
                            if residualizedControls and controlValues and withLanguage: 
                                gnListIndices = gnListIndices[:-1]
                            elif not withLanguage:
                                gnListIndices = [gnListIndices[-1]]
                            for i in gnListIndices:
                                groupNormValues = thisGroupNormsList[i]
                                #featureNames = featureNameList[i] #(if needed later, make sure to add controls to this)
                                (Xdicts, ydict) = (groupNormValues, outcomes)
                                print "   (feature group: %d): [Initial size: %d]" % (i, len(ydict))
                                (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                                (Xtest, ytest) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = testGroupsOrder)

                                assert len(ytest) == testSize, "ytest not the right size"
                                if len(ytrain) > self.trainingSize:
                                    Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                                num_feats += Xtrain.shape[1]
                                multiXtrain.append(Xtrain)
                                multiXtest.append(Xtest)
                            print "[Train size: %d    Test size: %d]" % (len(ytrain), len(ytest))

                            ################################
                            #4) fit model and test accuracy:
                            (regressor, multiScalers, multiFSelectors) = self._multiXtrain(multiXtrain, ytrain, standardize, sparse = sparse)
                            ypred = self._multiXpredict(regressor, multiXtest, multiScalers = multiScalers, multiFSelectors = multiFSelectors, sparse = sparse)
                            predictions.update(dict(izip(testGroupsOrder,ypred)))
                            #pprint(ypred[:10])
                                
                            ##4 a) save accuracy stats:
                              ## TODO: calculate all this at end instead
                            
                            R2 = metrics.r2_score(ytest, ypred)
                            mse = metrics.mean_squared_error(ytest, ypred)
                            mae = metrics.mean_absolute_error(ytest, ypred)
                            train_mean = mean(ytrain)
                            train_mean_mae = metrics.mean_absolute_error(ytest, [train_mean]*len(ytest))
                            print "  *FOLD R^2: %.4f (MSE: %.4f; MAE: %.4f; mean train mae: %.4f)"% (R2, mse, mae, train_mean_mae)
                            testStats['R2_folds'].append(R2)
                            (pearsr, r_p) = pearsonr(ytest, ypred)
                            testStats['r_folds'].append(pearsr)
                            testStats['r_p_folds'].append(r_p)
                            testStats['train_mean_mae_folds'].append(train_mean_mae)
                            testStats['mse_folds'].append(mse)
                            testStats['mae_folds'].append(mae)
                            testStats.update({'train_size': len(ytrain), 'test_size': len(ytest), 'num_features' : num_feats, 
                             '{model_desc}': str(regressor).replace('\t', "  ").replace('\n', " ").replace('  ', " "),
                             '{modelFS_desc}': str(multiFSelectors[0]).replace('\t', "  ").replace('\n', " ").replace('  ', " "),
                                              })
                            ##4 b) weighted eval
                            if wOutcome:
                                weights = [float(wOutcome[k]) for k in testGroupsOrder]
                                try:
                                    results = sm.WLS(zscore(ytest), zscore(ypred), weights).fit() #runs regression
                                    testStats['rwghtd_folds'].append(results.params[-1])
                                    testStats['rwghtd_p_folds'].append(results.pvalues[-1])
                                    #print results.summary(outcomeName, [outcomeName+'_pred'])#debug
                                except ValueError as err:
                                    print "WLS threw ValueError: %s\nresult not included" % str(err)

                        #########################
                        #5) aggregate test stats:
                        ## 5a) average fold results
                        reportStats = dict()
                        for k, v in testStats.items():
                            if isinstance(v, list):
                                reportStats[k] = mean(v)
                                reportStats['se_'+k] = std(v) / sqrt(float(nFolds))
                            else:
                                reportStats[k] = v

                        ## 5b) calculate overall accuracies
                        ytrue, ypred = ([], [])
                        if residualizedControls and controlValues and withLanguage:
                            ytrue, yres, yconpred, yrespred = alignDictsAsy(nonresOutcomes, outcomes, resControlPreds, predictions)
                            n = len(ytrue)
                            ypred = array(yrespred) + array(yconpred) #control predictions
                            for metric, value in self.accuracyStats(yres, yrespred).iteritems():
                                reportStats['resid_'+metric] = value
                            #compute paired t-test:
                            yfinalpred_res_abs = absolute(array(ytrue) - array(ypred))
                            yconpred_res_abs = absolute(array(ytrue) - array(yconpred))
                            yfcra_diff = yfinalpred_res_abs - yconpred_res_abs
                            yfcra_diff_mean, yfcra_sd = mean(yfcra_diff), std(yfcra_diff)
                            yfcra_diff_t = yfcra_diff_mean / (yfcra_sd / sqrt(n))
                            yfcra_diff_p = t.sf(np.abs(yfcra_diff_t), n-1)
                            (reportStats['paired_ttest_t'], reportStats['paired_ttest_p1tail']) =(yfcra_diff_t, yfcra_diff_p)
                            print "ttest from scratch: (%.4f, %.4f)"% (yfcra_diff_t, yfcra_diff_p)
                            #print "ttest from stats:   (%.4f, %.4f)"% ttest_1samp(yfcra_diff, 0)

                            #compute p-value based on fisher's z-transform:
                            (rfinal, _), (rcond, _) = pearsonr(ytrue, ypred), pearsonr(ytrue, yconpred)
                            (rfplus, rfminus) = ((rfinal+1), (1 - rfinal))
                            (rcplus, rcminus) = ((rcond+1), (1-rcond))
                            zfinal = (log(rfplus)-log(rfminus))/2
                            zcond = (log(rcplus)-log(rcminus))/2
                            #print "zfinal: %.6f" % zfinal
                            #print "zcond: %.6f" % zcond
                            se = sqrt((1.0/(n-3))+(1.0/(n-3)))
                            z = (zfinal-zcond)/se;
                            #print "se: %.6f" % se
                            #print "z: %.6f" % z
                            z2 = abs(z);
                            p2tail =(((((.000005383*z2+.0000488906)*z2+.0000380036)*z2+.0032776263)*z2+.0211410061)*z2+.049867347)*z2+1;
                            #print "p2tail-1: %.6f" % p2tail 
                            p2tail = p2tail**-16
                            #print "p2tail-2: %.6f" % p2tail 
                            p1tail = p2tail/2;
                            (reportStats['fisher_z_z'], reportStats['fisher_z_p1tail']) = (z, p1tail)

                        else:
                            ytrue, ypred = alignDictsAsy(outcomes, predictions)
                        reportStats.update(self.accuracyStats(ytrue, ypred))
                        reportStats['N'] = len(ytrue)

                        ## 5c) print overall stats
                        print "*Overall R^2:          %.4f" % (reportStats['R2'])
                        print "*Overall FOLDS R^2:    %.4f (+- %.4f)"% (reportStats['R2_folds'], reportStats['se_R2_folds'])
                        print "*R (sqrt R^2):         %.4f"% reportStats['R']
                        print "*Pearson r:            %.4f (p = %.5f)"% (reportStats['r'], reportStats['r_p'])
                        print "*Folds Pearson r:      %.4f (p = %.5f)"% (reportStats['r_folds'], reportStats['r_p_folds'])
                        if wOutcome:
                            print "*weighted lest squares:%.4f (p = %.5f)"% (reportStats['rwghtd_folds'], reportStats['rwghtd_p_folds'])
                        print "*Spearman rho:         %.4f (p = %.5f)"% (reportStats['rho'], reportStats['rho_p'])
                        print "*Mean Squared Error:   %.4f"% reportStats['mse']
                        print "*Mean Absolute Error:  %.4f"% reportStats['mae']
                        print "*Train_Mean MAE:       %.4f"% reportStats['train_mean_mae']
                        if 'paired_ttest_t' in reportStats:
                            print "*Paired T-test p:       %.5f (t: %.4f)"% (reportStats['paired_ttest_p1tail'], reportStats['paired_ttest_t'])
                            print "*Fisher r-to-z p:       %.5f (z: %.4f)"% (reportStats['fisher_z_p1tail'], reportStats['fisher_z_z'])


                        if savePredictions: 
                            reportStats['predictions'] = predictions
                        if saveModels: 
                            print "!!SAVING MODELS NOT IMPLEMENTED FOR testControlCombos!!"
                        try:
                            scores[outcomeName][controlKeyCombo][withLanguage] = reportStats
                        except KeyError:
                            scores[outcomeName][controlKeyCombo] = {withLanguage: reportStats}

        print "\n[TEST COMPLETE]\n"
        return scores

    def adjustOutcomesFromControls(self, standardize = True, sparse = False, saveModels = False, allControlsOnly = False, comboSizes = None, nFolds = 2, savePredictions = True, groupsWhere = ''):
        """Produces adjusted outcomes given the controls"""
        
        ###################################
        #1. setup groups for random folds
        (groups, allOutcomes, allControls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        print "[number of groups: %d (%d Folds)]" % (len(groups), nFolds)
        random.seed(self.randomState)
        groupList = sorted(list(groups), reverse=True)
        random.shuffle(groupList)
        groupFolds =  [x for x in foldN(groupList, nFolds)]
        XGroups = groups
        
        ################################
        #2b) setup control combinations:
        controlKeys = allControls.keys()
        scores = dict() #outcome => control_tuple => [0],[1] => scores= {R2, R, r, r-p, rho, rho-p, MSE, train_size, test_size, num_features,yhats}
        if not comboSizes:
            comboSizes = xrange(1, len(controlKeys)+1)
            if allControlsOnly:
                comboSizes = [len(controlKeys)]
        for r in comboSizes:
            for controlKeyCombo in combinations(controlKeys, r):
                controls = dict()
                if len(controlKeyCombo) > 0:
                    controls = dict([(k, allControls[k]) for k in controlKeyCombo])
                controlKeyCombo = tuple(controlKeyCombo)
                print "\n\n|COMBO: %s|" % str(controlKeyCombo)
                print '='*(len(str(controlKeyCombo))+9)
                controlValues = controls.values() #list of dictionaries of group=>group_norm

                #########################################
                #3. train / test models for each possible y:
                for outcomeName, outcomes in sorted(allOutcomes.items()):
                    thisOutcomeGroups = set(outcomes.keys()) & XGroups
                    if not outcomeName in scores:
                        scores[outcomeName] = dict()

                    print "\n= %s (NO lang.)=\n%s"%(outcomeName, '-'*(len(outcomeName)+14))
                    testStats = {'R2_folds': [], 'r_folds': [], 'r_p_folds': [], 'mse_folds': [], 'mae_folds': [], 'train_mean_mae_folds': []}

                    predictions = {}

                    #################################3
                    ## 4 calculate residuals
                    nonresOutcomes = dict(outcomes) #backup for accuracy calc. #new outcomes become residuals
                    resControlPreds = {} #predictions form controls only

                        #TODO: make this and below a function:
                    print "CREATING RESIDUALS"
                    #creating residuals:
                    for testChunk in xrange(0, len(groupFolds)):
                        print " Residual fold %d " % (testChunk)
                        trainGroups = set()
                        for chunk in (groupFolds[:testChunk]+groupFolds[(testChunk+1):]):
                            for c in chunk:
                                trainGroups.add(c)
                        testGroups = set(groupFolds[testChunk])
                        #set static group order across features:
                        trainGroupsOrder = list(thisOutcomeGroups & trainGroups)
                        testGroupsOrder = list(thisOutcomeGroups & testGroups)
                        testSize = len(testGroupsOrder)

                        (Xdicts, ydict) = (controlValues, outcomes)
                        print "  [Initial size: %d]" % (len(ydict))
                        (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                        (Xtest, ytest) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = testGroupsOrder)

                        assert len(ytest) == testSize, "ytest not the right size"
                        if len(ytrain) > self.trainingSize:
                            Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                        (res_regressor, res_multiScalers, res_multiFSelectors) = self._multiXtrain([Xtrain], ytrain, standardize, sparse = sparse)
                        res_ypred = self._multiXpredict(res_regressor, [Xtest], multiScalers = res_multiScalers, multiFSelectors = res_multiFSelectors, sparse = sparse)
                        #DEBUG: random sort
                        #random.shuffle(res_ypred)
                        resControlPreds.update(dict(izip(testGroupsOrder,res_ypred)))

                        R2 = metrics.r2_score(ytest, res_ypred)
                        mse = metrics.mean_squared_error(ytest, res_ypred)
                        mae = metrics.mean_absolute_error(ytest, res_ypred)
                        train_mean = mean(ytrain)
                        train_mean_mae = metrics.mean_absolute_error(ytest, [train_mean]*len(ytest))
                        print "  residual fold R^2: %.4f (MSE: %.4f; MAE: %.4f; mean train mae: %.4f)"% (R2, mse, mae, train_mean_mae)

                    #update outcomes to be residuals:
                    meanOutcome = mean(outcomes.values())
                    for gid, value in outcomes.items():
                        try: 
                            #print "true outcome: %.4f, controlPred: %.4f" % (value, resControlPreds[gid])#debug
                            outcomes[gid] = (value - resControlPreds[gid]) + meanOutcome
                            #print " new value %.4f" % outcomes[gid] #debug
                        except KeyError:
                            print " warn: no control prediction found for gid %s, removing from outcomes" % str(gid)
                            del outcomes[gid]
                    print "DONE CREATING RESIDUALS for %s %s\n" % (outcomeName, str(controlKeyCombo))

                    ## 5b) calculate overall accuracies
                    ytrue, yres, yconpred = alignDictsAsy(nonresOutcomes, outcomes, resControlPreds)
                    n = len(ytrue)
                    reportStats = self.accuracyStats(ytrue, yconpred)

                    ## 5c) print overall stats
                    print "*Overall R^2:          %.4f" % (reportStats['R2'])
                    print "*R (sqrt R^2):         %.4f"% reportStats['R']
                    print "*Pearson r:            %.4f (p = %.5f)"% (reportStats['r'], reportStats['r_p'])
                    print "*Spearman rho:         %.4f (p = %.5f)"% (reportStats['rho'], reportStats['rho_p'])
                    print "*Mean Squared Error:   %.4f"% reportStats['mse']
                    print "*Mean Absolute Error:  %.4f"% reportStats['mae']
                    print "*Train_Mean MAE:       %.4f"% reportStats['train_mean_mae']

                    if savePredictions: 
                        reportStats['predictions'] = outcomes
                    if saveModels: 
                        print "!!SAVING MODELS NOT IMPLEMENTED FOR testControlCombos!!"
                    try:
                        scores[outcomeName][controlKeyCombo][0] = reportStats
                    except KeyError:
                        scores[outcomeName][controlKeyCombo] = {0: reportStats}

        print "\n[TEST COMPLETE]\n"
        return scores



    def accuracyStats(self, ytrue, ypred):
        if not isinstance(ytrue[0], float):
            ytrue = [float(y) for y in ytrue]
        if not isinstance(ypred[0], float):
            ypred = [float(y) for y in ypred]
        reportStats = dict()
        reportStats['R2'] = metrics.r2_score(ytrue, ypred)
        reportStats['R'] = sqrt(reportStats['R2'])
        reportStats['mse'] = metrics.mean_squared_error(ytrue, ypred)
        reportStats['mae'] = metrics.mean_absolute_error(ytrue, ypred)
        train_mean = mean(ytrue)
        reportStats['train_mean_mae'] = metrics.mean_absolute_error(ypred, [train_mean]*len(ypred))
        (reportStats['r'], reportStats['r_p']) = pearsonr(ytrue, ypred)
        (reportStats['rho'], reportStats['rho_p']) = spearmanr(ytrue, ypred)
        return reportStats


    #################################################
    #################################################

    def predict(self, standardize = True, sparse = False, restrictToGroups = None, groupsWhere = ''):
        if not self.multiXOn:
            print "\n!! model trained without multiX, reverting to old predict !!\n"
            return self.old_predict(standardize, sparse, restrictToGroups)

        ################
        #1. setup groups
        (groups, allOutcomes, allControls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)

        if restrictToGroups: #restrict to groups
            rGroups = restrictToGroups
            if isinstance(restrictToGroups, dict):
                rGroups = [item for sublist in restrictToGroups.values() for item in sublist]
            groups = groups.intersection(rGroups)

            for outcomeName, outcomes in allOutcomes.iteritems():
                allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
            for controlName, controlValues in allControls.iteritems():
                allControls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % (len(groups))

        ####################
        #2. get data for Xs:
        groupNormsList = []
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        for i in xrange(len(self.featureGetters)):
            fg = self.featureGetters[i]
            (groupNorms, newFeatureNames) = fg.getGroupNormsSparseFeatsFirst(groups)

            print " [Aligning current X with training X: feature group: %d]" %i
            groupNormValues = []
            #print self.featureNamesList[i][:10]#debug
            for feat in self.featureNamesList[i]:
                if feat in groupNorms:
                    groupNormValues.append(groupNorms[feat])
                else:
                    groupNormValues.append(dict())
            groupNormsList.append(groupNormValues)
            print "  Features Aligned: %d" % len(groupNormValues)

            print " Groups in featureTable:", len(getGroupsFromGroupNormValues(groupNormValues)), "groups that have outcomes:", len(groups)
            if not XGroups:
                XGroups = set(getGroupsFromGroupNormValues(groupNormValues))
            else:
                # print "Maarten ", str(XGroups)[:300]
                XGroups = XGroups & getGroupsFromGroupNormValues(groupNormValues) #intersect groups from all feature tables
                # potential source of bug: if a sparse feature table doesn't have all of the groups which it should
        #XGroups = XGroups & groups #this should be needed
        if len(XGroups) < len(groups): 
            print " Different number of groups available for different outcomes (or feature tables)."
            print "   %d groups -> %d groups" % (len(groups), len(XGroups))
        
        ################################
        #2b) setup control data:
        controlValues = allControls.values()
        if controlValues:
            groupNormsList.append(controlValues)

        #########################################
        #3. predict for all possible outcomes
        predictions = dict()
        testGroupsOrder = list(XGroups) 
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            if isinstance(restrictToGroups, dict): #outcome specific restrictions:
                outcomes = dict([(g, o) for g, o in outcomes.iteritems() if g in restrictToGroups[outcomeName]])
            thisTestGroupsOrder = [gid for gid in testGroupsOrder if gid in outcomes]
            if len(thisTestGroupsOrder) < len(testGroupsOrder):
                print "   this outcome has less groups. Shrunk groups to %d total." % (len(thisTestGroupsOrder))
            (multiXtest, ytest) = ([], None)
            for i in xrange(len(groupNormsList)): #get each feature group data (i.e. feature table)
                groupNormValues = groupNormsList[i]
                (Xdicts, ydict) = (groupNormValues, outcomes)
                print "  (feature group: %d)" % (i)
                (Xtest, ytest) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = thisTestGroupsOrder)
                multiXtest.append(Xtest)
                print "   [Test size: %d ]" % (len(ytest))

            #############
            #4) predict
            ypred = self._multiXpredict(self.regressionModels[outcomeName], multiXtest, multiScalers = self.multiScalers[outcomeName], \
                                            multiFSelectors = self.multiFSelectors[outcomeName], sparse = sparse)
            print "[Done. Evaluation:]"
            R2 = metrics.r2_score(ytest, ypred)
            print "*R^2 (coefficient of determination): %.4f"% R2
            print "*R (sqrt R^2):                       %.4f"% sqrt(R2)
            print "*Pearson r:                          %.4f (%.5f)"% pearsonr(ytest, ypred)
            print "*Spearman rho:                       %.4f (%.5f)"% spearmanr(ytest, ypred)
            mse = metrics.mean_squared_error(ytest, ypred)
            print "*Mean Squared Error:                 %.4f"% mse
            mae = metrics.mean_absolute_error(ytest, ypred)
            print "*Mean Absolute Error:                %.4f"% mae
            assert len(thisTestGroupsOrder) == len(ypred), "can't line predictions up with groups" 
            predictions[outcomeName] = dict(zip(thisTestGroupsOrder,ypred))

        print "[Prediction Complete]"

        return predictions

    def predictToOutcomeTable(self, standardize = True, sparse = False, fe = None, name = None, nFolds = 10):

        # step1: get groups from feature table
        groups = self.featureGetter.getDistinctGroupsFromFeatTable()
        groups = list(groups)
        chunks = [groups]

        # 2: chunks of groups (only if necessary)
        if len(groups) > self.maxPredictAtTime:
            numChunks = int(len(groups) / float(self.maxPredictAtTime)) + 1
            print "TOTAL GROUPS (%d) TOO LARGE. Breaking up to run in %d chunks." % (len(groups), numChunks)
            random.seed(self.randomState)
            random.shuffle(groups)        
            chunks =  [x for x in foldN(groups, numChunks)]

        predictions = dict() #outcomes->groups->prediction               

        totalPred = 0
        for c,chunk in enumerate(chunks):
            print "\n\n**CHUNK %d\n" % c

            # 3: predict for each chunk for each outcome
            
            chunkPredictions = self.predictNoOutcomeGetter(chunk, standardize, sparse)
            #predictions is now outcomeName => group_id => value (outcomeName can become feat)
            #merge chunk predictions into predictions:
            for outcomeName in chunkPredictions.iterkeys():
                try:
                    predictions[outcomeName].update(chunkPredictions[outcomeName])
                except KeyError:
                    predictions[outcomeName] = chunkPredictions[outcomeName]
            if chunk:
                totalPred += len(chunk)
            else: totalPred = len(groups)
            print " Total Predicted: %d" % totalPred

        featNames = predictions.keys()
        predDF = pd.DataFrame(predictions)
        # print predDF

        name = "p_%s" % self.modelName[:4] + "$" + name
        # 4: use self.outcomeGetter.createOutcomeTable(tableName, dataFrame)
        self.outcomeGetter.createOutcomeTable(name, predDF, 'replace')

    def predictNoOutcomeGetter(self, groups, standardize = True, sparse = False, restrictToGroups = None):
        
        outcomes = list(self.regressionModels.keys())

        ####################
        #2. get data for Xs:
        groupNormsList = []
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        UGroups = None
        for i in xrange(len(self.featureGetters)):
            fg = self.featureGetters[i]
            (groupNorms, newFeatureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            
            print " [Aligning current X with training X: feature group: %d]" %i
            groupNormValues = []
            
            for feat in self.featureNamesList[i]:
                # groupNormValues is a list in order of featureNamesList of all the group_norms
                groupNormValues.append(groupNorms.get(feat, {}))

            groupNormsList.append(groupNormValues)
            print "  Features Aligned: %d" % len(groupNormValues)

            fgGroups = getGroupsFromGroupNormValues(groupNormValues)
            # fgGroups has diff nb cause it's a chunk

            if not XGroups:
                XGroups = set(fgGroups)
                UGroups = set(fgGroups)
            else:
                XGroups = XGroups & fgGroups #intersect groups from all feature tables
                UGroups = UGroups | fgGroups #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should
                #TODO: fill these in with zeros but print an obvious warning because it could also be a sign of 
                #      non-english messages which aren't triggering any feautres
        #XGroups = XGroups & groups #this should not be needed
        if len(XGroups) < len(UGroups): 
            print " !! Different number of groups available for different feature tables. (%d, %d)\n this may cause problems down the line" % (len(XGroups), len(groups))
        

        #########################################
        #3. predict for all possible outcomes
        predictions = dict()
        #testGroupsOrder = list(UGroups) #use the union of feature tables (may cause null issue)
        testGroupsOrder = list(XGroups)#use the intersection only
        
        for outcomeName in sorted(outcomes):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            thisTestGroupsOrder = testGroupsOrder

            multiXtest = []
            for i in xrange(len(groupNormsList)): #get each feature group data (i.e. feature table)
                
                groupNormValues = groupNormsList[i] # basically a feature table in list(dict) form
                # We want the dictionaries to turn into a list that is aligned
                gns = dict(zip(self.featureNamesList[i], groupNormValues))
                # print "Maarten", str(gns)[:150]
                df = pd.DataFrame(data=gns)
                df = df.fillna(0.0)
                df = df[self.featureNamesList[i]]
                df = df.reindex(thisTestGroupsOrder)
                print "  (feature group: %d)" % (i)

                multiXtest.append(csr_matrix(df.values))
                # print "Maarten", csr_matrix(df.values).shape, csr_matrix(df.values).todense()

            #############
            #4) predict
            ypred = self._multiXpredict(self.regressionModels[outcomeName], multiXtest, multiScalers = self.multiScalers[outcomeName], \
                                            multiFSelectors = self.multiFSelectors[outcomeName], sparse = sparse)
            print "[Done.]"

            assert len(thisTestGroupsOrder) == len(ypred), "can't line predictions up with groups" 
            predictions[outcomeName] = dict(zip(thisTestGroupsOrder,ypred))

        print "[Prediction Complete]"

        return predictions



    def predictAllToFeatureTable(self, standardize = True, sparse = False, fe = None, name = None, nFolds = 10, groupsWhere = ''):

        print "\n !! Mostly sure but not completely that this will produce same resutls as combo_test_reg !! \n"

        if not fe:
            print "Must provide a feature extractor object"
            sys.exit(0)

        #1. get all groups
        
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        
        #split groups into n chunks (same as comboControlTest)
        random.seed(self.randomState)
        groupList = sorted(list(groups), reverse=True)
        random.shuffle(groupList)
        groupFolds =  [x for x in foldN(groupList, nFolds)]
        chunks =  [x for x in foldN(groupList, nFolds)]
        predictions = dict() #outcomes->groups->prediction               
        #for loop over testChunks, changing which one is teh test (not test = training)
        for testChunk in xrange(0, len(chunks)):
            trainGroups = set()
            for chunk in (chunks[:testChunk]+chunks[(testChunk+1):]):
                for c in chunk:
                    trainGroups.add(c)
            testGroups = set(chunks[testChunk])
            self.train(standardize, sparse, restrictToGroups = trainGroups)
            chunkPredictions = self.predict(standardize, sparse, testGroups )
            #predictions is now outcomeName => group_id => value (outcomeName can become feat)
            #merge chunk predictions into predictions:
            for outcomeName in allOutcomes.keys():
                if outcomeName in predictions:
                    predictions[outcomeName].update(chunkPredictions[outcomeName]) ##check if & works
                else:
                    predictions[outcomeName] = chunkPredictions[outcomeName]

        #output to table:
        featNames = predictions.keys()

        featLength = max(map(lambda s: len(s), featNames))

        #CREATE TABLE:
        featureName = "p_%s" % self.modelName[:4]
        if name: featureName += '_' + name
        featureTableName = fe.createFeatureTable(featureName, "VARCHAR(%d)"%featLength, 'DOUBLE')

        #write predictions to database (no need for "REPLACE" because we are creating the table)
        for feat in featNames:
            preds = predictions[feat]

            print "[Inserting Predictions as Feature values for %s]" % feat
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, '"""+feat+"""', %s, %s)"""
            rows = [(k, v, v) for k, v in preds.iteritems()] #adds group_norm and applies freq filter
            mm.executeWriteMany(fe.corpdb, fe.dbCursor, wsql, rows, writeCursor=fe.dbConn.cursor(), charset=fe.encoding, use_unicode=fe.use_unicode)

    def predictToFeatureTable(self, standardize = True, sparse = False, fe = None, name = None, groupsWhere = ''):
        if not fe:
            print "Must provide a feature extractor object"
            sys.exit(0)

        # handle large amount of predictions:
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)

        if len(allOutcomes) == 0:
            print """
      ERROR: predictToFeatureTable doesn't work when --outcome_table
             and --outcomes are not specified, please make a dummy
             table containing zero-ed outcome columns
             """
            sys.exit(0)

        groups = list(groups)
        # groups contains all groups that are in the outcome table and that have outcome values not null
        
        chunks = [groups]
        #split groups into chunks
        if len(groups) > self.maxPredictAtTime:
            random.seed(self.randomState)
            random.shuffle(groups)        
            chunks =  [x for x in foldN(groups, int(ceil(len(groups) / self.maxPredictAtTime)))]
        predictions = dict()
        totalPred = 0

        featNames = None
        featLength = None
        featureTableName = ""

        for chunk in chunks:
            if len(chunk) == len(groups):
                print "len(chunk) == len(groups): True"
                # chunk = None

            chunkPredictions = self.predict(standardize, sparse, chunk)

            # predictions is now outcomeName => group_id => value (outcomeName can become feat)
            # merge chunk predictions into predictions:
            for outcomeName in chunkPredictions.iterkeys():
                try:
                    predictions[outcomeName].update(chunkPredictions[outcomeName])
                except KeyError:
                    predictions[outcomeName] = chunkPredictions[outcomeName]
            totalPred += len(chunk)
            print " Total Predicted: %d" % totalPred
            
            # INSERTING the chunk into MySQL
            #Creating table if not done yet
            if not featNames:
                featNames = chunkPredictions.keys()
                featLength = max(map(lambda s: len(s), featNames))
                # CREATE TABLE, and insert into it progressively:
                featureName = "p_%s" % self.modelName[:4]
                if name: featureName += '_' + name
                featureTableName = fe.createFeatureTable(featureName, "VARCHAR(%d)"%featLength, 'DOUBLE')

            written = 0
            rows = []
            for feat in featNames:
                preds = chunkPredictions[feat]
                
                print "[Inserting Predictions as Feature values for feature: %s]" % feat
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, '"""+feat+"""', %s, %s)"""
                
                for k, v in preds.iteritems():
                    rows.append((k, v, v))
                    if len(rows) >  self.maxPredictAtTime or len(rows) >= len(preds):
                        mm.executeWriteMany(fe.corpdb, fe.dbCursor, wsql, rows, writeCursor=fe.dbConn.cursor(), charset=fe.encoding, use_unicode=fe.use_unicode)
                        written += len(rows)
                        print "   %d feature rows written" % written
                        rows = []
            # if there's rows left
            if rows:
                mm.executeWriteMany(fe.corpdb, fe.dbCursor, wsql, rows, writeCursor=fe.dbConn.cursor(), charset=fe.encoding, use_unicode=fe.use_unicode)
                written += len(rows)
                print "   %d feature rows written" % written
        return
                
                

        # write n-grams to database (no need for "REPLACE" because we are creating the table)
        # for feat in featNames:
        #    preds = predictions[feat]

            #print "[Inserting Predictions as Feature values for %s]" % feat
            #wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, '"""+feat+"""', %s, %s)"""
            #rows = []
            #written = 0
            #for k, v in preds.iteritems():
             #   rows.append((k, v, v))
             #   if len(rows) >  self.maxPredictAtTime or len(rows) >= len(preds):
             #       fe._executeWriteMany(wsql, rows)
             #       written += len(rows)
             #       print "   %d feature rows written" % written
             #       rows = []
           # if rows:
           #     fe._executeWriteMany(wsql, rows)
           #     written += len(rows)
           #     print "   %d feature rows written" % written
           #     rows = []

    def getWeightsForFeaturesAsADict(self): 
        #returns dictionary of featureTable -> outcomes -> feature_name -> weight
        #weights: self.regressionModels[outcome].coef_
        # Scaler: self.multiScalers[outcome]
        weights_dict = dict()

        featTables = [fg.featureTable for fg in self.featureGetters]
        # MAARTEN
        # pprint(self.regressionModels)
        # pprint([feat[:20] for feat in self.featureNamesList])
        startIndex = 0 # for multiX regression
        just_one_feat = None
        for i, featTableFeats in enumerate(self.featureNamesList):

            weights_dict[featTables[i]] = dict()
            for outcome, model in self.regressionModels.iteritems():
                weights_dict[featTables[i]][outcome] = dict()
                coefficients = eval('self.regressionModels[outcome].%s' % self.modelToCoeffsName[self.modelName.lower()])
                # print len(coefficients) # debug
                intercept = self.regressionModels[outcome].intercept_
                print "Intercept [" + str(intercept) + "]"
                # print "coefficients size: %s " % str(coefficients.shape)

                # Getting the right chunk if there are multiple feature tables (OLD)
                # coefficients = coefficients[startIndex:(startIndex+len(featTableFeats))]
                
                
                
                coefficients.resize(1,len(coefficients))
                # Inverting Feature Selection
                
                if self.multiFSelectors[outcome][i]:
                    print "Inverting the feature selection: %s" % self.multiFSelectors[outcome][i]
                    coefficients = self.multiFSelectors[outcome][i].inverse_transform(coefficients)#.flatten()
                
                if 'mean_' in dir(self.multiFSelectors[outcome][i]):
                    print "RPCA mean: ", self.multiFSelectors[outcome][i].mean_

                if 'steps' in dir(self.multiFSelectors[outcome][i]):
                    if 'mean_' in dir(self.multiFSelectors[outcome][i].steps[1][1]):
                        print self.multiFSelectors[outcome][i].steps[1][1].mean_, len(self.multiFSelectors[outcome][i].steps[1][1].mean_)
                        mean = self.multiFSelectors[outcome][i].steps[1][1].mean_.copy()
                        t = self.multiFSelectors[outcome][i].steps[0][1].inverse_transform(mean).flatten()
                        print self.multiFSelectors[outcome][i].steps[1][1].transform(mean-intercept), self.multiFSelectors[outcome][i].steps[1][1].transform(mean-intercept).sum()
                        print t, len(t)
                        print coefficients
                        # coefficients = coefficients + t
                        print coefficients
                        # INTERCEPT CORRECTION : dot product of coefficients.mean from PCA
                        print "Pipelines don't work with this option (predtolex)"
                        sys.exit()


                # coefficients.resize(1,len(coefficients))
                # print coefficients.shape
                # Inverting the scaling 
                if self.multiScalers[outcome][i]:
                    print "Standardscaler doesn't work with Prediction To Lexicon"
                    print self.multiScalers[outcome][i].mean_, self.multiScalers[outcome][i].std_
                    coefficients = self.multiScalers[outcome][i].inverse_transform(coefficients).flatten()
                else: 
                    coefficients = coefficients.flatten() 
                
                # featTableFeats contains the list of features 
                if len(coefficients) != len(featTableFeats):
                    print "length of coefficients (%d) does not match number of features (%d)" % (len(coefficients), len(featTableFeats))
                    sys.exit(1)
                # print len(featTableFeats), len(coefficients), coefficients.shape # debug

                weights_dict[featTables[i]][outcome] = {featTableFeats[j]: coefficients[j] for j in xrange(len(featTableFeats)) if coefficients[j] != 0}
                weights_dict[featTables[i]][outcome]['_intercept'] = intercept

                # just_one_feat = {featTableFeats[j]: coefficients[j] for j in xrange(len(featTableFeats))}

                #with open("ageLex.pickle",'w+') as a:
                #    pickle.dump(just_one_feat, a)
                #print len(weights_dict[featTables[i]][outcome])

            startIndex += len(featTableFeats)

        #pprint(weights_dict.items()[0][1].items()[0][1].items()[1])

        return weights_dict

        #return just_one_feat

    def _train(self, X, y, standardize = True):
        """does the actual regression training, first feature selection: can be used by both train and test"""

        sparse = True
        if not isinstance(X, csr_matrix):
            X = np.array(X)
            sparse = False
        scaler = None
        if standardize == True:
            scaler = StandardScaler(with_mean = not sparse)
            print "[Applying StandardScaler to X: %s]" % str(scaler)
            X = scaler.fit_transform(X)
        y = np.array(y)
        print " (N, features): %s" % str(X.shape)

        fSelector = None
        if self.featureSelectionString and X.shape[1] >= self.featureSelectMin:
            fSelector = eval(self.featureSelectionString)
            if self.featureSelectPerc < 1.0:
                print "[Applying Feature Selection to %d perc of X: %s]" % (int(self.featureSelectPerc*100), str(fSelector))
                _, Xsub, _, ysub = train_test_split(X, y, test_size=self.featureSelectPerc, train_size=1, random_state=0)
                fSelector.fit(Xsub, ysub)
                newX = fSelector.transform(X)
                if newX.shape[1]:
                    X = newX
                else:
                    print "No features selected, so using original full X"
            else:
                print "[Applying Feature Selection to X: %s]" % str(fSelector)
                newX = fSelector.fit_transform(X, y)
                if newX.shape[1]:
                    X = newX
                else:
                    print "No features selected, so using original full X"
            print " after feature selection: (N, features): %s" % str(X.shape)

        modelName = self.modelName.lower()
        if (X.shape[1] / float(X.shape[0])) < self.backOffPerc: #backoff to simpler model:
            print "number of features is small enough, backing off to %s" % self.backOffModel
            modelName = self.backOffModel.lower()

        if hasMultValuesPerItem(self.cvParams[modelName]) and modelName[-2:] != 'cv':
            #grid search for classifier params:
            gs = GridSearchCV(eval(self.modelToClassName[modelName]+'()'), 
                              self.cvParams[modelName], n_jobs = self.cvJobs)
            print "[Performing grid search for parameters over training]"
            gs.fit(X, y, cv=ShuffleSplit(len(y), n_iterations=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))

            print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
            return gs.best_estimator_, scaler, fSelector
        else:
            # no grid search
            print "[Training regression model: %s]" % modelName
            regressor = eval(self.modelToClassName[modelName]+'()')
            regressor.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
            #print dict(self.cvParams[modelName][0])

            regressor.fit(X, y)
            #print "coefs"
            #print regressor.coef_
            print "model: %s " % str(regressor)
            if modelName[-2:] == 'cv' and 'alphas' in regressor.get_params():
                print "  selected alpha: %f" % regressor.alpha_
            return regressor, scaler, fSelector


    def _multiXtrain(self, X, y, standardize = True, sparse = False):
        """does the actual regression training, first feature selection: can be used by both train and test
           craete multiple scalers and feature selectors
           and just one regression model (of combining the Xes into 1)
        """

        if not isinstance(X, (list, tuple)):
            X = [X]
        multiX = X
        X = None #to avoid errors
        multiScalers = []
        multiFSelectors = []

        for i in xrange(len(multiX)):
            X = multiX[i]
            if not sparse:
                X = X.todense()

            #Standardization:
            scaler = None
            if standardize == True:
                scaler = StandardScaler(with_mean = not sparse)
                print "[Applying StandardScaler to X[%d]: %s]" % (i, str(scaler))
                X = scaler.fit_transform(X)
                y = np.array(y)
            print " X[%d]: (N, features): %s" % (i, str(X.shape))

            #Feature Selection
            fSelector = None
            if self.featureSelectionString and X.shape[1] >= self.featureSelectMin:
                fSelector = eval(self.featureSelectionString)
                if self.featureSelectPerc < 1.0:
                    print "[Applying Feature Selection to %d perc of X: %s]" % (int(self.featureSelectPerc*100), str(fSelector))
                    _, Xsub, _, ysub = train_test_split(X, y, test_size=self.featureSelectPerc, train_size=1, random_state=0)
                    fSelector.fit(Xsub, ysub)
                    newX = fSelector.transform(X)
                    if newX.shape[1]:
                        X = newX
                    else:
                        print "No features selected, so using original full X"
                else:
                    print "[Applying Feature Selection to X: %s]" % str(fSelector)
                    newX = fSelector.fit_transform(X, y)
                    if newX.shape[1]:
                        X = newX
                    else:
                        print "No features selected, so using original full X"
                print " after feature selection: (N, features): %s" % str(X.shape)

            multiX[i] = X
            multiScalers.append(scaler)
            multiFSelectors.append(fSelector)      

        #combine all multiX into one X:
        X = multiX[0]
        for nextX in multiX[1:]:
            X = np.append(X, nextX, 1)
        modelName = self.modelName.lower()
        if (X.shape[1] / float(X.shape[0])) < self.backOffPerc: #backoff to simpler model:
            print "number of features is small enough, backing off to %s" % self.backOffModel
            modelName = self.backOffModel.lower()

        if hasMultValuesPerItem(self.cvParams[modelName]) and modelName[-2:] != 'cv':
            #grid search for classifier params:
            gs = GridSearchCV(eval(self.modelToClassName[modelName]+'()'), 
                              self.cvParams[modelName], n_jobs = self.cvJobs)
            print "[Performing grid search for parameters over training]"
            #gs.fit(X, y, cv=ShuffleSplit(len(y), n_iterations=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))
            gs.fit(X, y)

            print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
            return gs.best_estimator_,  multiScalers, multiFSelectors
        else:
            # no grid search
            print "[Training regression model: %s]" % modelName
            regressor = eval(self.modelToClassName[modelName]+'()')
            try: 
                regressor.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
            except IndexError: 
                print "No CV parameters available"
                raise IndexError
            #print dict(self.cvParams[modelName][0])

            try:
                regressor.fit(X, y)
            except LinAlgError:
                print "Lin Algebra error, X:"
                pprint(X)
                
    
            print "model: %s " % str(regressor)
            if modelName[-2:] == 'cv' and 'alphas' in regressor.get_params():
                print "  selected alpha: %f" % regressor.alpha_
            return regressor, multiScalers, multiFSelectors

    
    def _predict(self, regressor, X, scaler = None, fSelector = None, y = None):
        if scaler:
            X = scaler.transform(X)
        if fSelector:
            newX = fSelector.transform(X)
            if newX.shape[1]:
                X = newX
            else:
                print "No features selected, so using original full X"

        return regressor.predict(X)

    def _multiXpredict(self, regressor, X, multiScalers = None, multiFSelectors = None, y = None, sparse = False):
        if not isinstance(X, (list, tuple)):
            X = [X]

        multiX = X
        X = None #to avoid errors

        for i in xrange(len(multiX)):

            #setup X and transformers:
            X = multiX[i]
            if not sparse:
                X = X.todense()
            (scaler, fSelector) = (None, None)
            if multiScalers: 
                scaler = multiScalers[i]
            if multiFSelectors:
                fSelector = multiFSelectors[i]

            #run transformations:
            if scaler:
                print "  predict: applying standard scaler to X[%d]: %s" % (i, str(scaler)) #debug
                X = scaler.fit_transform(X)
            if fSelector:
                print "  predict: applying feature selection to X[%d]: %s" % (i, str(fSelector)) #debug
                newX = fSelector.transform(X)
                if newX.shape[1]:
                    X = newX
                else:
                    print "No features selected, so using original full X"
            multiX[i] = X

        #combine all multiX into one X:
        X = multiX[0]
        for nextX in multiX[1:]:
            X = np.append(X, nextX, 1)

        print "  predict: combined X shape: %s" % str(X.shape) #debu
        if hasattr(regressor, 'intercept_'):
            print "  predict: regression intercept: %f" % regressor.intercept_

        return regressor.predict(X)


    ######################
    def load(self, filename):
        print "[Loading %s]" % filename
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        print "Outcomes in loaded model:", tmp_dict['regressionModels'].keys()
        tmp_dict['outcomes'] = tmp_dict['regressionModels'].keys() 
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        print "[Saving %s]" % filename
        f = open(filename,'wb')
        toDump = {'modelName': self.modelName, 
                  'regressionModels': self.regressionModels,
                  'multiScalers': self.multiScalers,
                  'scalers' : self.scalers, 
                  'multiFSelectors': self.multiFSelectors,
                  'fSelectors' : self.fSelectors,
                  'featureNames' : self.featureNames,
                  'featureNamesList' : self.featureNamesList,
                  'multiXOn' : self.multiXOn,
                  'controlsOrder' : self.controlsOrder
                  }
        pickle.dump(toDump,f,2)
        f.close()

    ########################
    @staticmethod 
    def printComboControlScoresToCSV(scores, outputstream = sys.stdout, paramString = None, delimiter='|'):
        """prints scores with all combinations of controls to csv)"""
        if paramString: 
            print >>outputstream, paramString+"\n"
        i = 0
        outcomeKeys = sorted(scores.keys())
        previousColumnNames = []
        for outcomeName in outcomeKeys:
         
            outcomeScores = scores[outcomeName]
            #setup column and row names:
            controlNames = sorted(list(set([controlName for controlTuple in outcomeScores.keys() for controlName in controlTuple])))
            rowKeys = sorted(outcomeScores.keys(), key = lambda k: len(k))
            scoreNames = sorted(list(set(name for k in rowKeys for v in outcomeScores[k].itervalues() for name in v.keys())), key=str.lower)
            #scoreNames = sorted(outcomeScores[rowKeys[0]].itervalues().next().keys(), key=str.lower)
            columnNames = ['row_id', 'outcome', 'model_controls'] + scoreNames + ['w/ lang.'] + controlNames

            #csv:
            csvOut = csv.DictWriter(outputstream, fieldnames=columnNames, delimiter=delimiter)
            if set(columnNames) != set(previousColumnNames):
                firstRow = dict([(str(k), str(k)) for k in columnNames if k is not 'predictions'])
                csvOut.writerow(firstRow)
                previousColumnNames = columnNames
            for rk in rowKeys:
                for withLang, sc in outcomeScores[rk].iteritems():
                    i+=1
                    rowDict = {'row_id': i, 'outcome': outcomeName, 'model_controls': str(rk)+str(withLang)}
                    for cn in controlNames:
                        rowDict[cn] = 1 if cn in rk else 0
                    rowDict['w/ lang.'] = withLang
                    rowDict.update({(k,v) for (k,v) in sc.items() if k is not 'predictions'})
                    csvOut.writerow(rowDict)
    @staticmethod
    def printComboControlPredictionsToCSV(scores, outputstream, paramString = None, delimiter='|'):
        """prints predictions with all combinations of controls to csv)"""
        predictionData = {}
        data = defaultdict(list)
        columns = ["Id"]
        if paramString:
            print >>outputstream, paramString+"\n"
        i = 0
        outcomeKeys = sorted(scores.keys())
        previousColumnNames = []
        for outcomeName in outcomeKeys:
            outcomeScores = scores[outcomeName]
            controlNames = sorted(list(set([controlName for controlTuple in outcomeScores.keys() for controlName in controlTuple])))
            rowKeys = sorted(outcomeScores.keys(), key = lambda k: len(k))
            for rk in rowKeys:
                for withLang, s in outcomeScores[rk].iteritems():
                    i+=1
                    mc = "_".join(rk)
                    if(withLang):
                        mc += "_withLanguage"
                    columns.append(outcomeName+'_'+mc)
                    predictionData[str(i)+'_'+outcomeName+'_'+mc] = s['predictions']
                    for k,v in s['predictions'].iteritems():
                        data[k].append(v)
        
        writer = csv.writer(outputstream)
        writer.writerow(columns)
        for k,v in data.iteritems():
           v.insert(0,k)  
           writer.writerow(v)
        
    #################
    ## Deprecated:
    def old_train(self, standardize = True, sparse = False, restrictToGroups = None):
        """Trains regression models"""
        #if restrictToGroups is a dict than it is an outcome-specific restriction

        print
        #1. get data possible ys (outcomes)
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        if restrictToGroups: #restrict to groups
            rGroups = restrictToGroups
            if isinstance(restrictToGroups, dict):
                rGroups = [item for sublist in restrictToGroups.values() for item in sublist]
            groups = groups.intersection(rGroups)
            for outcomeName, outcomes in allOutcomes.iteritems():
                allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
            for controlName, controlValues in controls.iteritems():
                controls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % len(groups)

        #2. get data for X:
        (groupNorms, featureNames) = (None, None)
        if len(self.featureGetters) > 1: 
            print "WARNING: multiple feature tables passed in rp.train only handles one for now."
        if sparse:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsSparseFeatsFirst(groups)
        else:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)

        self.featureNames = groupNorms.keys() #holds the order to expect features
        groupNormValues = groupNorms.values() #list of dictionaries of group => group_norm
        controlValues = controls.values() #list of dictionaries of group=>group_norm
    #     this will return a dictionary of dictionaries


        #3. Create classifiers for each possible y:
        for outcomeName, outcomes in allOutcomes.iteritems():
            if isinstance(restrictToGroups, dict): #outcome specific restrictions:
                outcomes = dict([(g, o) for g, o in outcomes.iteritems() if g in restrictToGroups[outcomeName]])
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            print "[Aligning Dicts to get X and y]"
            (X, y) = alignDictsAsXy(groupNormValues+controlValues, outcomes, sparse)
            (self.regressionModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName]) = self._train(X, y, standardize)

        print "[Done Training All Outcomes]"

    def old_predict(self, standardize = True, sparse = False, restrictToGroups = None):
        #predict works with all sklearn models
        #1. get data possible ys (outcomes)
        print
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        if restrictToGroups: #restrict to groups
            rGroups = restrictToGroups
            if isinstance(restrictToGroups, dict):
                rGroups = [item for sublist in restrictToGroups.values() for item in sublist]
            groups = groups.intersection(rGroups)
            for outcomeName, outcomes in allOutcomes.iteritems():
                allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
            for controlName, controlValues in controls.iteritems():
                controls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % len(groups)

        #2. get data for X:
        (groupNorms, featureNames) = (None, None)
        if len(self.featureGetters) > 1: 
            print "WARNING: multiple feature tables passed in rp.predict only handles one for now."
        if sparse:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsSparseFeatsFirst(groups)
        else:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)

        #reorder group norms:
        groupNormValues = []
        predictGroups = None
        print "[Aligning current X with training X]"
        for feat in self.featureNames:
            if feat in groupNorms:
                groupNormValues.append(groupNorms[feat])
            else:
                if sparse: #assumed all zeros
                    groupNormValues.append(dict())
                else: #need to insert 0s
                    if not predictGroups:
                        predictGroups = groupNorms[groupNorms.iterkeys().next()].keys()
                    groupNormValues.append(dict([(k, 0.0) for k in predictGroups]))
        #groupNormValues = groupNorms.values() #list of dictionaries of group => group_norm
        print "number of features after alignment: %d" % len(groupNormValues)
        controlValues = controls.values() #list of dictionaries of group=>group_norm (TODO: including controls for predict might mess things up)
    #     this will return a dictionary of dictionaries

        #3. Predict ys for each model:
        predictions = dict() #outcome=>group_id=>value
        for outcomeName, outcomes in allOutcomes.iteritems():
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            if isinstance(restrictToGroups, dict): #outcome specific restrictions:
                outcomes = dict([(g, o) for g, o in outcomes.iteritems() if g in restrictToGroups[outcomeName]])
            (X, ytest, keylist) = alignDictsAsXy(groupNormValues+controlValues, outcomes, sparse, returnKeyList = True)
            leny = len(ytest)
            print "[Groups to predict: %d]" % leny
            (regressor, scaler, fSelector) = (self.regressionModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName])
            ypred = None
            if self.chunkPredictions:
                ypred = np.array([]) #chunks:
                for subX, suby in chunks(X, ytest, self.maxPredictAtTime):
                    ysubpred = self._predict(regressor, subX, scaler = scaler, fSelector = fSelector)
                    ypred = np.append(ypred, np.array(ysubpred))
                    print "   num predicticed: %d" % len(ypred)
            else:
                ypred = self._predict(regressor, X, scaler = scaler, fSelector = fSelector)
            print "[Done Predicting. Evaluating]"
            R2 = metrics.r2_score(ytest, ypred)
            print "*R^2 (coefficient of determination): %.4f"% R2
            print "*R (sqrt R^2):                       %.4f"% sqrt(R2)
            print "*Pearson r:                          %.4f (%.5f)"% pearsonr(ytest, ypred)
            print "*Spearman rho:                       %.4f (%.5f)"% spearmanr(ytest, ypred)
            mse = metrics.mean_squared_error(ytest, ypred)
            print "*Mean Squared Error:                 %.4f"% mse
            predictions[outcomeName] = dict(zip(keylist,ypred))

        return predictions


            
####################################################################
##
#
class CombinedRegressionPredictor(RegressionPredictor):
    """A class to handle a combination of regression predictors, implemented as a linear model"""
    #cross validation parameters:
    cvParams = { #just for the combined model (change in regression predictor to change for the others)
        'ridge': [
            {'alpha': [10], 'fit_intercept':[True]}, #
            ],
        'ridgecv': [
            {'alphas': np.array([10, 2, 20, 1, 100, 200, 1000, .1, .2, .01, .001, .0001, .00001, .000001])},
            ],
       }
    modelToClassName = {
        'ridge' : 'Ridge',
        'ridgecv' : 'RidgeCV',
        }
    cvJobs = 6 #normal
    cvFolds = 3
    maxPredictAtTime = DEFAULT_MAX_PREDICT_AT_A_TIME 
    maxPredictAtTime = maxPredictAtTime*100 #DELETE THIS - MAARTEN

    testPerc = .20 #percentage of sample to use as test set (the rest is training)
    combinedTrainPerc = .15 #percentage of training data to hold out for the combined method
    randomState = 42 #random state when seeding something

    def __init__(self, og, fgs, modelNames = ['ridge'], combinedModelName = 'ridgecv'):
        #initialize combined regression predictor vars:
        self.outcomeGetter = og 
        self.modelName = combinedModelName
        self.combinedModels = []#stores combined models (not implemented yet)
        #no scaler needed.. ys from other models shoudl be the same self.scalers = dict()
        #no feature selection for combined regression: self.fSelectors = dict()
        self.featureSelectionString = None
        #self.featureNames = [] #holds the order the features are expected in

        #sub-predictors variables
        self.regressionPredictors = list()
        if not isinstance(modelNames, list):
            modelNames = [modelNames]
        if len(modelNames) != len(fgs):
            modelNames = [modelNames[0]] * len(fgs)
        print modelNames
        print fgs
        for i in xrange(len(fgs)):
            self.regressionPredictors.append(RegressionPredictor(og, fgs[i], modelNames[i]))


    def train(self, standardize = True, sparse = False):
        """Trains regression models"""
        raise NotImplementedError

    def test(self, standardize = True, sparse = False, saveModels = False, groupsWhere = ''):
        """Tests combined regression"""

        #1. get data possible ys (outcomes)
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        print "[combined regression: number of groups: %d]" % len(groups)
        
        #2: pick trainModels/traincombined / test split based only on groups:
        testSize = int(self.testPerc*len(groups))
        groups = list(groups)
        random.seed(self.randomState)
        random.shuffle(list(groups))
        (groupsTrain, groupsTest) = (groups[testSize:], groups[:testSize])
        #separate training set for individual models versus combined models:
        combTrainSize = int(self.combinedTrainPerc*len(groupsTrain)) #should already be shuffled
        (groupsTrainIndiv, groupsTrainComb) = (groupsTrain[combTrainSize:], groupsTrain[:combTrainSize])
        print "[Overall Train size: %d (%d indiv, %d combined) Test size: %d]" % (len(groupsTrain), len(groupsTrainIndiv), len(groupsTrainComb), len(groupsTest))

        #3 train each rp on the selected groups
        print "[Training Individual Classifiers]"
        for rp in self.regressionPredictors:
            rp.train(standardize, sparse, restrictToGroups = groupsTrainIndiv)

        #4: get X info for combined method (predict on individual models):
        modelTrainPreds = list()
        modelTestPreds = list()
        for i in xrange(len(self.regressionPredictors)):
            rp = self.regressionPredictors[i]
            print "\n[Getting Prediction Data for Combined X from %s]" % rp.featureGetter.featureTable
            modelTrainPreds.append(rp.predict(standardize, sparse, restrictToGroups = groupsTrainComb))
            print "[Test results for individual feature sets:]"
            modelTestPreds.append(rp.predict(standardize, sparse, restrictToGroups = groupsTest))


        #5: train/test regressors for each possible y:
        print "\n\n***Combined Results***\n"
        controlValues = controls.values()
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            for num_feats in range(2, len(self.regressionPredictors)+1):
                for combo in combinations(range(len(self.regressionPredictors)), num_feats):
                    print "\n = Combo: %s" % ', '.join([self.regressionPredictors[i].featureGetter.featureTable for i in combo])

                    #produce X, ys: (assuming columns are aligned...)
                    modelPs = [modelTrainPreds[i][outcomeName] for i in combo]
                    (Xtrain, ytrain) = alignDictsAsXy(modelPs+controlValues, outcomes, False)
                    modelTestPs = [modelTestPreds[i][outcomeName] for i in combo]
                    (Xtest, ytest) = alignDictsAsXy(modelTestPs+controlValues, outcomes, False)
                    print "  [Combined Train Size: %d, Test size: %d]" % (len(ytrain), len(ytest))

                    (regressor, scaler, _) = self._train(Xtrain, ytrain, standardize)
                    ypred = self._predict(regressor, Xtest, scaler = scaler) #never use fselector for combined
                    R2 = metrics.r2_score(ytest, ypred)
                    #R2 = r2simple(ytest, ypred)
                    print "  *R^2 (coefficient of determination): %.4f"% R2
                    print "  *R (sqrt R^2):                       %.4f"% sqrt(R2)
                    print "  *Pearson r:                          %.4f (p = %.5f)"% pearsonr(ytest, ypred)
                    print "  *Spearman rho:                       %.4f (p = %.5f)"% spearmanr(ytest, ypred)
                    mse = metrics.mean_squared_error(ytest, ypred)
                    print "  *Mean Squared Error:                 %.4f"% mse

        print "\n[COMBINATION TEST COMPLETE]\n"

    def predict(self, standardize = True, testPerc = 0.25, sparse = False):
        #predict works with all sklearn models
        raise NotImplementedError
         
    def predictToFeatureTable(self, standardize = True, testPerc = 0.25, sparse = False, fe = None, name = None):
        raise NotImplementedError
            

    ######################
    def load(self, filename):
        print "[Loading %s]" % filename
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    def save(self, filename):
        print "[Saving %s]" % filename
        f = open(filename,'wb')
        toDump = {'modelName': self.modelName, 
                  'regressionModels': self.regressionModels,
                  'scalers' : self.scalers, 
                  'fSelectors' : self.fSelectors,
                  'featureNames' : self.featureNames
                  }
        pickle.dump(toDump,f,2)
        f.close()


####################################################################
##
#
DEF_KEEP_CLASSES = set([1, -1])

class ClassifyToRegressionPredictor:

    #Vars:
    randomState = 42 #random state when seeding something
    testPerc = .20 #percentage of sample to use as test set (the rest is training)

    classOutcomeLabel = 'bin_'

    """Performs classificaiton for 0/non-zero then regression on non-zeros"""
    def __init__(self, og, fg, modelC = 'linear-svc', modelR = 'ridgecv'):
        #initialize the predictors: ogC and ogR should have same groups
        #outcome table must have self.classOutcomeLabel version of each outcome
        #create ogC from og:
        ogC = og.copy()
        ogC.outcome_value_fields= [self.classOutcomeLabel+o for o in ogC.outcome_value_fields]
        self.classifyPredictor = ClassifyPredictor(ogC, fg, modelC)
        self.regressionPredictor = RegressionPredictor(og, fg, modelR)
        self.keepClasses = DEF_KEEP_CLASSES

    def train(self, standardize = True, sparse = False, restrictToGroups = None, nFolds = 4, trainRegOnAll = True, classifierAsFeat = True, groupsWhere = ''):

        #1. get groups for both:
        print "[ClassifytoRegression: Getting all Groups]"            
        (classifyGroups, _, _) = self.classifyPredictor.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        (regressionGroups, _, _) = self.regressionPredictor.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        if restrictToGroups:
            classifyGroups = set(classifyGroups) & set(restrictToGroups)
            regressionGroups = set(regressionGroups) & set(restrictToGroups)
        groups = list(set(classifyGroups) & set(regressionGroups))
        if len(groups) < len(classifyGroups) or len(groups) < len(regressionGroups):
            print >> sys.stderr, "intersecting groups not equal to original sizes. cGroups: %d, rGroups: %d, intersection: %d" % \
                (len(classifyGroups), len(regressionGroups), len(groups))
        random.seed(self.randomState)
        random.shuffle(groups)

        #2. use nFold cross-validation to train / predict every portion for classification
        print "[ClassifytoRegression: Training / Predicting with Classifiers to get Regressor Groups]"            
        restrictedTrainGroups = dict() #outcme->list of group_ids
        if not trainRegOnAll:
            subGroups = [x for x in foldN(groups, nFolds)]
            for foldNum in xrange(len(subGroups)):
                trainGroups = [item for s in subGroups[0:foldNum] + subGroups[foldNum+1:] for item in s]
                predictGroups = subGroups[foldNum]
                self.classifyPredictor.train(standardize, sparse, restrictToGroups = trainGroups)
                outPreds = self.classifyPredictor.predict(standardize, sparse, restrictToGroups = predictGroups)
                for outcome, ypreds in outPreds.iteritems():
                    newKeeps = [g for g, c in ypreds.iteritems() if c in self.keepClasses]
                    outcome = outcome[len(self.classOutcomeLabel):]
                    try:
                        restrictedTrainGroups[outcome] = restrictedTrainGroups[outcome].union(newKeeps)
                    except KeyError:
                        restrictedTrainGroups[outcome] = set(newKeeps)

        #3. train the classifier on all data
        print "[ClassifytoRegression: Training Full Classifiers]"            
        self.classifyPredictor.train(standardize, sparse, restrictToGroups = groups)

        #4. train regression on all keeps from step2
        print "[ClassifytoRegression: Training Regressors]"
        if trainRegOnAll:
                self.regressionPredictor.train(standardize, sparse, restrictToGroups = groups)
        else:
            self.regressionPredictor.train(standardize, sparse, restrictToGroups = restrictedTrainGroups)

        print "[DONE TRAINING C2R MODELS]"
        

    def test(self, standardize = True, sparse = False, saveModels = False, groupsWhere = ''):
        #1. get groups for both
        print "[ClassifytoRegression: Getting all Groups]"            
        (classifyGroups, _, _) = self.classifyPredictor.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        (regressionGroups, _, _) = self.regressionPredictor.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        groups = list(set(classifyGroups) & set(regressionGroups))
        if len(groups) < len(classifyGroups) or len(groups) < len(regressionGroups):
            print >> sys.stderr, "intersecting groups not equal to original sizes. cGroups: %d, rGroups: %d, intersection: %d" % \
                (len(classifyGroups), len(regressionGroups), len(groups))

        #2. random shuffle and choose train / test set
        random.seed(self.randomState)
        random.shuffle(groups)
        testSize = int(self.testPerc*len(groups))
        (groupsTrain, groupsTest) = (groups[testSize:], groups[:testSize])

        #3. run train
        self.train(standardize, sparse, restrictToGroups=groupsTrain)

        #4. run predict on held-out
        self.predict(standardize, sparse, restrictToGroups=groupsTest)

    def predict(self, standardize = True, sparse = False, restrictToGroups = None):
        """Predicts with the classifier and regressor. zero must be the false prediction from classifier """
        #1. run classifier
        print "[C2R: Predicting with classifier]"
        cPreds = self.classifyPredictor.predict(standardize, sparse, restrictToGroups = restrictToGroups, groupsWhere = '')

        #2. pick groups where classified as keeps
        restrictRegGroups = dict()
        #if self.classAsFeature
        for outcome, ypreds in cPreds.iteritems():
            newKeeps = [g for g, c in ypreds.iteritems() if c in self.keepClasses]
            outcome = outcome[len(self.classOutcomeLabel):]
            try:
                restrictRegGroups[outcome] = restrictRegGroups[outcome].union(newKeeps)
            except KeyError:
                restrictRegGroups[outcome] = set(newKeeps)

        #3. predict with regression
        print "[C2R: Predicting non-zeros with Regression]"
        print len(restrictRegGroups)
        print restrictRegGroups.keys()
        rPreds = self.regressionPredictor.predict(standardize, sparse, restrictToGroups = restrictRegGroups)

        #4. add in zeros for others
        for outcome, ypreds in rPreds.iteritems():
            cGroups = cPreds[self.classOutcomeLabel+outcome].keys()
            for g in cGroups:
                if g not in ypreds:
                    ypreds[g] = 0

        #5. evaluate
        print "[C2R: Done Predicting. Evaluating Overall]"
        (groups, outcomes, controls) = self.regressionPredictor.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        for outcome, yhats in rPreds.iteritems():
            print "\n= %s =\n%s"%(outcome, '-'*(len(outcome)+4))
            ytest, ypred = alignDictsAsy(outcomes[outcome], yhats)
            R2 = metrics.r2_score(ytest, ypred)
            print "*R^2 (coefficient of determination): %.4f"% R2
            print "*R (sqrt R^2):                       %.4f"% sqrt(R2)
            print "*Pearson r:                          %.4f (%.5f)"% pearsonr(ytest, ypred)
            print "*Spearman rho:                       %.4f (%.5f)"% spearmanr(ytest, ypred)
            mse = metrics.mean_squared_error(ytest, ypred)
            print "*Mean Squared Error:                 %.4f"% mse

        return rPreds
       

class RPCRidgeCV(LinearModel, RegressorMixin):
    """Randomized PCA Ridge Regression with built-in cross-validation

    To set the RPCA number of components, it uses a train(80%) and dev-set(20%). 

    For ridge, it performs Generalized Cross-Validation, which is a form of
    efficient Leave-One-Out cross-validation.

    Parameters
    ----------
    component_percs: list of percentages
        number of components to try as a percentage of observations
        default is  [0.01, 0.0333, 0.1, 0.333, 1]

    alphas: numpy array of shape [n_alphas]
        Array of alpha values to try.
        Small positive values of alpha improve the conditioning of the
        problem and reduce the variance of the estimates.
        Alpha corresponds to ``(2*C)^-1`` in other linear models such as
        LogisticRegression or LinearSVC.

    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional
        If True, the regressors X are normalized

    cv : cross-validation generator, optional
        If None, Generalized Cross-Validation (efficient Leave-One-Out)
        will be used.

    gcv_mode : {None, 'auto', 'svd', eigen'}, optional
        Flag indicating which strategy to use when performing
        Generalized Cross-Validation. Options are::

            'auto' : use svd if n_samples > n_features, otherwise use eigen
            'svd' : force computation via singular value decomposition of X
            'eigen' : force computation via eigendecomposition of X^T X

        The 'auto' mode is the default and is intended to pick the cheaper \
        option of the two depending upon the shape of the training data.

    store_cv_values : boolean, default=False
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the `cv_values_` attribute (see
        below). This flag is only compatible with `cv=None` (i.e. using
        Generalized Cross-Validation).

    Attributes
    ----------
    `cv_values_` : array, shape = [n_samples, n_alphas] or \
        shape = [n_samples, n_targets, n_alphas], optional
        Cross-validation values for each alpha (if `store_cv_values=True` and \
        `cv=None`). After `fit()` has been called, this attribute will \
        contain the mean squared errors (by default) or the values of the \
        `{loss,score}_func` function (if provided in the constructor).

    `coef_` : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    `alpha_` : float
        Estimated regularization parameter.

    See also
    --------
    RidgeCV: Ridge Regression with built-in cross-val to set alpha
    Ridge: Ridge regression
    """

    #the reduction technique to use (must have n_comps):
    reducerString = 'RandomizedPCA(n_components=n_comps, random_state=42, whiten=False, iterated_power=3)'
    #reducerString = 'Pipeline([("1_rpca", RandomizedPCA(n_components=n_comps, random_state=42, whiten=False, iterated_power=3)), ("2_univariate_select", SelectFpr(f_regression, alpha=0.1))])'

    def __init__(self, component_percs = [0.01, 0.0316, 0.1, 0.316, 1],
                 alphas=np.array([0.1, 1.0, 10.0]),
                 fit_intercept=True, normalize=False, 
                 cv=None, gcv_mode=None):
        self.component_percs = sorted(component_percs)
        self.reducer = None #stores the chosen reducer
        self.estimator = None #stores the chosen ridgeCV estimator
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.cv = cv
        self.gcv_mode = gcv_mode

    def fit(self, X, y, sample_weight=1.0):
        """fits the randomized pca and ridge through cross-validation """
        component_percs = self.component_percs
        rows = X.shape[0]
        cols = X.shape[1]
        for j in xrange(len(component_percs)):
            if component_percs[j]*rows > int(cols/2):
                component_percs = component_percs[:j]
                break
        if len(component_percs) == 0:
            print " no componen percs were small enough, using cols/2"
            component_percs = array([int((cols/2)/rows)])

        errors = dict()
        min_error = float("inf")
        best_perc = 0
        print "seraching over %s" % str(component_percs)

        #search for best perc
        searched = set([]);
        i = 0
        last_error = float("inf")
        numErrIncrease = 0 #when reaches 2 it will break
        while i >= 0 and i < len(component_percs) and not i in searched:
            perc = component_percs[i]
            n_comps = int(rows * perc) #evaluated in reducerString to set components
            reducer =  eval(self.reducerString)
            print " RPCRidgeCV: Fitting reducer for crossval:%.4f\n %s" % (perc, str(reducer))
            reducer.fit(X, y)
            reducedX = reducer.transform(X)
            if not reducedX.shape[1]:
                reducedX = X
                print "  **No features selected, so using original full X"
            print " ReducedX: %s "  % str(reducedX.shape)

            #train ridge
            ridgeString = "RidgeCV(%s, %s, %s, None, None, %s, %s, store_cv_values=True)" \
                % (str(list(self.alphas)), str(self.fit_intercept), str(self.normalize), str(self.cv), str(self.gcv_mode))
            ridgeCV = eval(ridgeString)
            ridgeCV.fit(reducedX, y, sample_weight)
            error = min([min(vals) for vals in ridgeCV.cv_values_])
            if error < min_error:
                min_error = error
                best_perc = perc

            errors[perc] = {'error':error, 'estimator': ridgeCV, 'reducer': reducer}

            #compare with previous to decide next i:
            searched.add(i)
            if last_error <= error:#doing worse, take the last
                numErrIncrease +=1
            if numErrIncrease >= 2:
                break
            i += 1
            last_error = error

        pprint(errors)
        print "Selected %.4f" % best_perc
        self.reducer = errors[best_perc]['reducer']
        self.estimator = errors[best_perc]['estimator']
        self.alpha_ = self.estimator.alpha_

        return self

    def transform(self, X):
        return self.reducer.transform(X)

    def predict(self, X):
        X = self.transform(X)
        return self.estimator.predict(X)


####################################################################
##
#
class VERPCA(RandomizedPCA):
    """Randomized PCA that sets number of components by variance explained

    Parameters
    ----------
    n_components : int
        Maximum number of components to keep: default is 50.

    copy : bool
        If False, data passed to fit are overwritten

    iterated_power : int, optional
        Number of iteration for the power method. 3 by default.

    whiten : bool, optional
        When True (False by default) the `components_` vectors are divided
        by the singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

    max_components_ratio : float
        Maximum number of components in terms of their ratio to the number 
        of features. Default is 0.25 (1/4). 
    """

    #Vars


    def __init__(self, n_components=None, copy=True, iterated_power=3,
                 whiten=False, random_state=None, max_components_ratio = 0.25):
        if n_components > 0 and n_components < 1:
            self.variance_explained = n_components
            n_components = None
        else:
            self.variance_explained = None 
        self.max_components_ratio = max_components_ratio
        super(VERPCA, self).__init__(n_components, copy, iterated_power, whiten, random_state)

    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X: array-like or scipy.sparse matrix, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.n_components is None:
            self.n_components = min(int(X.shape[1]*self.max_components_ratio), X.shape[1])

        super(VERPCA, self).fit(X, y)

        #reduce to variance explained:
        if self.variance_explained:
            totalV = 0
            i = 0
            for i in xrange(len(self.explained_variance_ratio_)):
                totalV += self.explained_variance_ratio_[i]
                if (i%10 == 0) or (i < 10):
                    # print "%d: %f" % (i, totalV) #debug
                    pass
                if totalV >= self.variance_explained:
                    i += 1
                    break
            self.n_components = i

            #change components matrix (X)
            self.components_ = self.components_[:i]

        return self

         

class SpamsGroupLasso(LinearModel, RegressorMixin):
    """interfaces with the spams implementation of group lasso"""

    #Vars
    spamsParams = {'numThreads' : -1,
                   'verbose' : True,
                   }
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 rho=None):

        raise NotImplementedError

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if rho is not None:
            self.l1_ratio = rho
            warnings.warn("rho was renamed to l1_ratio and will be removed "
                          "in 0.15", DeprecationWarning)
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0



#########################
## Module Methods

def chunks(X, y, size):
    """ Yield successive n-sized chunks from X and Y."""
    if not isinstance(X, csr_matrix):
        assert len(X) == len(y), "chunks: size of X and y don't match"
    size = max(len(y), size)
    
    for i in xrange(0, len(y), size):
        yield X[i:i+size], y[i:i+size]


def grouper(folds, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    n = len(l) / folds
    return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def foldN(l, folds):
    """ Yield successive n-sized chunks from l."""
    n = len(l) / folds
    last = len(l) % folds
    for i in xrange(0, n*folds, n):
        if i+n+last >= len(l):
            yield l[i:i+n+last]
            break
        else: 
            yield l[i:i+n]

def hasMultValuesPerItem(listOfD):
    """returns true if the dictionary has a list with more than one element"""
    if len(listOfD) > 1:
        return True
    for d in listOfD:
        for value in d.itervalues():
            if len(value) > 1: return True
    return False

def getGroupsFromGroupNormValues(gnvs):
    return set([k for gns in gnvs for k in gns.iterkeys()])


#def mean(l):
#    return sum(l)/float(len(l))

def r2simple(ytrue, ypred):
    y_mean = sum(ytrue)/float(len(ytrue))
    ss_tot = sum((yi-y_mean)**2 for yi in ytrue)
    ss_err = sum((yi-fi)**2 for yi,fi in zip(ytrue,ypred))
    r2 = 1 - (ss_err/ss_tot)
    return r2
    ###Deprecated:
#     def trainLinearModel(self, groupFreqThresh = 0, standardize = True):
#         """Trains a linear regression model"""
        
#         #1. get data possible ys (outcomes)
#         (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupFreqThresh)
#   #     - this will give you 3 values: groups, outcomes, and controls
#         #       groups lists all of the groups (i.e. users) who meet groupFreqThresh criteria
#         #       outcomes is a dictionary(hashtable) of dictionaries
#         #       you can ignore controls
#         #       

#         #2. get data for X:
#         langFeatures = self.featureGetter.getGroupNormsWithZeros(groups)
#   #     this will return a dictionary of dictionaries

#         #3. Create predictors for each possible y:
#         predictors = dict() #store scikit-learn models here
#         scalers = dict()
#         for outcomeName, outcomes in allOutcomes.iteritems():
#            (X, y) = alignDictsAsXy(langFeatures[0], outcomes, list(langFeatures[1]))
#      X = np.array(X)
#      if standardize == True:
#       scalers[outcomeName] = preprocessing.StandardScaler()
#       X = scalers[outcomeName].fit_transform(X)

#            # X, y = shuffle(X, y, random_state=0) #if cross-validating and suspect data is not randomly ordered
#            #y = zscore(y) #debug: remove in future
#      n = len(y)
#      split = (n*4)/5
#      print 'Shape of X', X.shape
#      X_train = X[:split]
#      #print 'shape of X train', X_train.shape
#            X_test = X[split:]
#      #print 'shape of Xtest', X_test.shape
#      y = np.array(y)
#      y = y.reshape(n,1)
#      y_train = y[:split]
#            y_test = y[split:]
# #    selector = SelectPercentile(f_regression, percentile=10)
# #    X_train = selector.fit_transform(X_train, y_train)
# #    X_test = selector.transform(X_test)
# #    print 'Shape of feature selected x train', X_train.shape
# #    print 'Shape of feature selected x test', X_test.shape
#      predictors[outcomeName] = linear_model.Ridge(alpha=0.1)
#            predictors[outcomeName].fit(X_train,y_train)
           
#      #Debugging information
#            predictedY = predictors[outcomeName].predict(X_test)
#      #print "[%s] Mean error rate on test data: %.4f" % (outcomeName, predictors[outcomeName].score(X_test,y_test))
#      print "[%s] Mean error rate (R2) on test data: %.4f" % (outcomeName, metrics.r2_score(y_test, predictedY))
#      print "[%s] mean square error: %.4f" % (outcomeName, metrics.mean_square_error(y_test, predictedY))

#         #Set object's regressionModels and scalers
