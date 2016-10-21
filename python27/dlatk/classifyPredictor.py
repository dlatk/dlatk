#!/usr/bin/python
#########################################
# Classify Predictor
#
# Interfaces with FeatureWorker and scikit-learn
# to perform prediction of outcomes for lanaguage features.
#
# example: predicting satisfaction with life score given language use
#
# example usage: ./featureWorker.py --outcome_fields SWL --test_classifiers
#
# currently supported models: svc, linear-svc

import cPickle as pickle
#import pickle

from inspect import ismethod
import sys
import random
from itertools import combinations, izip_longest, izip
import csv
import time

from pprint import pprint
import collections

import pandas as pd 

from collections import defaultdict, Counter

#scikit-learn imports
from sklearn.svm import SVC, LinearSVC 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit, train_test_split
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score
from sklearn.feature_selection import f_classif, SelectPercentile, SelectKBest, SelectFdr, SelectFpr, SelectFwe

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier

from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit, train_test_split
from sklearn.decomposition import RandomizedPCA, MiniBatchSparsePCA, PCA, KernelPCA, NMF
from sklearn.lda import LDA #linear descriminant analysis
from sklearn.grid_search import GridSearchCV 
from sklearn import metrics

from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model.base import LinearModel
from sklearn.base import ClassifierMixin

#scipy
import scipy.stats as ss
from scipy.stats import zscore
from scipy.stats.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix, vstack, hstack, spmatrix
import numpy as np
from numpy import sqrt, array, std, mean, bincount, int64
import math

#For ROC curves
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#infrastructure
from mysqlMethods import mysqlMethods as mm


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

    # Keys: list of group_ids in order.

    if sparse:
        keyToIndex = dict([(keys[i], i) for i in xrange(len(keys))])
        row = []
        col = []
        data = []
        
        # Columns: features.

        for c in xrange(len(X)):
            column = X[c]
            for keyid, value in column.iteritems():
                if keyid in keyToIndex:
                    row.append(keyToIndex[keyid])
                    col.append(c)
                    data.append(value)
        sparseX = csr_matrix((data,(row,col)), shape = (len(keys), len(X)))
        if returnKeyList:
            return (sparseX, array(listy).astype(int64), keys)
            #return (sparseX, array(listy), keys)
        else:
            return (sparseX, array(listy).astype(int64))
            #return (sparseX, array(listy))
        
    else: 
        listX = map(lambda k: [x[k] for x in X], keys)
        if returnKeyList:
            return (array(listX), array(listy).astype(int64), keys)
            #return (array(listX), array(listy), keys)
        else:
            return (array(listX), array(listy).astype(int64))
            #return (array(listX), array(listy))

def alignDictsAsXyWithFeats(X, y,feats):
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


def pos_neg_auc(y1, y2):
    auc = 1.0
    try:
        auc = roc_auc_score(y1, y2)
    except ValueError as err:
        print "AUC threw ValueError: %s\nbogus auc included" % str(err)
    if auc < 0.5:
        auc = auc - 1
    return auc


# def alignDictsAsy(y, yhat, keys = None):
#     if not keys: 
#         keys = frozenset(y.keys())
#         keys = keys.intersection(yhat.keys())
#     listy = map(lambda k: y[k], keys)
#     listyhat = map(lambda k: yhat[k], keys)
#     return (listy, listyhat)

class ClassifyPredictor:
    """Handles prediction of continuous outcomes"""

    #cross validation parameters:
    #cross validation parameters:
    cvParams = {
        'svc': [
            {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.0001, 0.1, 0.00001, 1], 'C': [1, 10, 100, 1000, 10000, 0.1]}, 
            #{'kernel': ['rbf'], 'gamma': [0.001, 0.0001, 0.00001], 'C': [1, 10, 100, 1000, 0.1]},  #msg-level
            #{'kernel': ['rbf'], 'gamma': [0.001, 0.0001, 0.01, .1, 0.00001, 0.000001], 'C': [10, 100, 1000, 10000]},  #msg-level timediffonly
            #{'kernel': ['rbf'], 'gamma': [0.0001, 0.00001, 0.000001], 'C': [10, 1, 100]},  #swl
            #{'kernel': ['rbf'], 'gamma': [0.0001, 0.00001, 0.000001], 'C': [10, 1, 100]},  #swl
            #{'kernel': ['rbf'], 'gamma': [0.00001], 'C': [10]}, #msg-level ppf
            #{'kernel': ['rbf']}, 
            #{'kernel': ['sigmoid'], 'C': [0.01, 0.001, .1, .0001, 10, 1]},                                        
            #{'kernel': ['linear'], 'C': [100, 10, 1000, 1], 'random_state': [42]}
            ],
        'linear-svc':[
            #{'C':[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0025, 0.00025, 0.00001, 0.000001, 0.000025, 0.0000001, 0.0000025, 0.00000001, 0.00000025], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]}, #swl
            #{'C':[100, 10, 1, 500, 1000, 5000, 10000, 25000, 0.1, 0.01, 0.001, 0.0001], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]}
            #{'C':[0.0001, 0.00001, 0.000001, 0.0000001], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]}
            #{'C':[0.01, 0.1, 0.001, 1, 0.0001, 10, 0.00001, 0.000001, 0.0000001], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]} #swl user
            #{'C':[0.01, 0.1, 0.001, 1, 0.0001, 10], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]} #depression user
            #{'C':[0.01, 0.1, 0.001, 0.0001, 0.00001, 0.000001], 'loss':['l2'], 'penalty':['l2'], 'dual':[False]} #depression user lots o feats
            #{'C':[1, 100, 10, 5, 2, 0.1], 'loss':['l2'], 'penalty':['l2'], 'dual':[True]}, #topics, personality
            #{'C':[100, 10, 1000, 10000], 'loss':['l1'], 'penalty':['l2'], 'dual':[True]},
            #{'C':[1000], 'loss':['l1'], 'penalty':['l2'], 'dual':[True]},
            #{'C':[100, 10, 1000, 10000], 'loss':['l2'], 'penalty':['l1'], 'dual':[False]},
            #{'C':[100, 10, 1000, 1, 10000], 'penalty':['l1'], 'dual':[False]},
            #{'C':[100, 10, 1000, 1, 10000, .1], 'penalty':['l1'], 'dual':[False]},
            #{'C':[100, 500, 10, 1000, 5000, 1, 10000], 'loss': ['l2'], 'penalty':['l2'], 'dual':[True]},
            #{'C':[5000]},
            #{'C':[100],'penalty':['l1'], 'dual':[False]} # 1gram gender classification no standardize no featureSelection
            #{'C':[100, 10, 1000], 'penalty':['l1'], 'dual':[False]},
            #{'C':1.0, 'fit_intercept':True, 'loss':'l2', penalty:'l2'}, #if not centered data, use fit_intercept
            #{'C':1.0, 'dual':False}, #when N_samples > n_features
            #{'C':[0.01, 0.1, 1, 10, 0.001], 'penalty':['l2'], 'dual':[True]} #timex message-level
            #{'C':[0.01], 'penalty':['l2'], 'dual':[True]} #timex message-level

            ###with l1 feature selection:
            #{'C':[0.01, 0.1, 0.001], 'penalty':['l1'], 'dual':[False], 'class_weight':['balanced']}, #FIRST PASS l1 OPTION; swl message-level
            #{'C':[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0025, 0.00025, 0.00001, 0.000001, 0.000025, 0.0000001, 0.0000025, 0.00000001, 0.00000025], 'penalty':['l1'], 'dual':[False]} #swl
            {'C':[0.01], 'penalty':['l1'], 'dual':[False]} #age, general sparse setting #words n phrases, gender (best 0.01=> 91.4 )
            #{'C':[0.1], 'penalty':['l1'], 'dual':[False], 'multi_class':['crammer_singer']} #
            #{'C':[0.01, 0.1, 1, 10, 0.001], 'penalty':['l1'], 'dual':[False]} #timex message-level
            #{'C':[0.000001], 'penalty':['l2']}, # UnivVsMultiv choice Maarten 
            #{'C':[0.001], 'penalty':['l1'], 'dual':[False]}, # UnivVsMultiv choice Maarten 
            #{'C':[1000000], 'penalty':['l2'], 'dual':[False]} #simulate l0
            #{'C':[1, 10, 0.1, 0.01, 0.05, 0.005], 'penalty':['l1'], 'dual':[False]} #swl/perma message-level
            ],
        'lr': [
            #{'C':[0.01, 0.1, 0.001, 1, .0001, 10], 'penalty':['l2'], 'dual':[False]}, 
            #{'C':[0.01, 0.1, 0.001, 1], 'penalty':['l2'], 'dual':[False]}, 
            #{'C':[.001], 'penalty':['l2'], 'dual':[False]},
            #{'C':[0.01, 0.1, 0.001, 1, .0001, 10], 'penalty':['l1'], 'dual':[False]},
            #{'C':[0.1, 1, 0.01], 'penalty':['l1'], 'dual':[False]} #timex message-level
            #{'C':[10, 1, 100, 1000], 'penalty':['l1'], 'dual':[False]} 
            #{'C':[0.01, 0.1, 0.001, 0.0001, 0.00001], 'penalty':['l1'], 'dual':[False]} #timex l2 rpca....
            #{'C':[0.1, 1, 10], 'penalty':['l1'], 'dual':[False]} #timex l2 rpca....
            #{'C':[0.00001], 'penalty':['l2']} # UnivVsMultiv choice Maarten 
            {'C':[0.01], 'penalty':['l1']} # UnivVsMultiv choice Maarten
            #{'C':[1000000], 'penalty':['l2'], 'dual':[False]} # for a l0 penalty approximation
            #{'C':[1000000000000], 'penalty':['l2'], 'dual':[False]} # for a l0 penalty approximation
            #{'C':[100], 'penalty':['l1'], 'dual':[False]} # gender prediction
            ],

        'etc': [ 
            #{'n_jobs': [10], 'n_estimators': [250], 'criterion':['gini']}, 
            #{'n_jobs': [10], 'n_estimators': [1000], 'criterion':['gini']}, 
            #{'n_jobs': [10], 'n_estimators': [100], 'criterion':['entropy']}, 
            #{'n_jobs': [12], 'n_estimators': [50], 'max_features': ["sqrt", "log2", None], 'criterion':['gini'], 'min_samples_split': [1]}, 
            #{'n_jobs': [12], 'n_estimators': [1000], 'max_features': ["sqrt"], 'criterion':['gini'], 'min_samples_split': [2]}, 
            {'n_jobs': [12], 'n_estimators': [200], 'max_features': ["sqrt"], 'criterion':['gini'], 'min_samples_split': [2]}, 
            {'n_jobs': [9], 'n_estimators': [1500], 'max_features': ["sqrt"], 'criterion':['gini'], 
             'min_samples_split': [2], 'class_weight': ['auto'], 'random_state': [42]}, 
            ],
        'etcsmall': [ 
            {'n_jobs': [8], 'n_estimators': [1500], 'max_features': [1.00], 'criterion':['gini'], 
             'min_samples_split': [2], 'class_weight': ['auto'], 'bootstrap': [True], 
             'random_state': [42], 'oob_score':[True]}, 
            ],
        'rfc': [

            #{'n_jobs': [10], 'n_estimators': [1000]}, 
            {'n_jobs': [8], 'n_estimators': [1500], 'max_features': ["sqrt"], 'criterion':['gini'], 
             'min_samples_split': [2], 'class_weight': ['auto'], 'bootstrap': [True], 
             'random_state': [42], 'oob_score':[True]}, 
            ],
        'pac': [
            {'n_jobs': [10], 'C': [1, .1, 10]}, 
            ],
        'lda': [
            {}, 
            ],
        'gbc': [
            {'n_estimators': [500], 'random_state': [42], 
             'subsample':[0.4], 'max_depth': [5]  },
            ],

        }

    modelToClassName = {
        'lr' : 'LogisticRegression',
        'linear-svc' : 'LinearSVC',
        'svc' : 'SVC',
        'etc' : 'ExtraTreesClassifier',
        'etcsmall' : 'ExtraTreesClassifier',
        'rfc' : 'RandomForestClassifier',
        'pac' : 'PassiveAggressiveClassifier',
        'lda' : 'LDA', #linear discriminant analysis
        'gbc' : 'GradientBoostingClassifier',
        }
    
    modelToCoeffsName = {
        ##TODO##
        'linear-svc' : 'coef_',
        'svc' : 'coef_',
        'lr': 'coef_'
        }

    #cvJobs = 3 #when lots of data 
    #cvJobs = 6 #normal
    cvJobs = 8 #resource-heavy

    cvFolds = 3
    chunkPredictions = False #whether or not to predict in chunks (good for keeping track when there are a lot of predictions to do)
    maxPredictAtTime = 150000
    backOffPerc = .00 #when the num_featrue / training_insts is less than this backoff to backoffmodel
    backOffModel = 'linear-svc'
    #backOffModel = 'lr'
    #backOffModel = 'linear'

    # feature selection:
    featureSelectionString = None
    #featureSelectionString = 'SelectKBest(f_classif, k=int(len(y)/3))'
    #featureSelectionString = 'SelectFdr(f_classif, alpha=0.01)' #THIS DOESNT SEEM TO WORK!
    #featureSelectionString = 'SelectFpr(f_classif)' #this is correlation feature selection
    #featureSelectionString = 'SelectFwe(f_classif, alpha=30.0)' #this is correlation feature selection w/ correction
    #featureSelectionString = 'SelectFwe(f_classif, alpha=200.0)'   ### TRY THIS ###
    #featureSelectionString = 'SelectPercentile(f_classif, 33)'#1/3 of features
    #featureSelectionString = 'ExtraTreesClassifier(n_jobs=10, n_estimators=100, compute_importances=True)'
    #featureSelectionString = \
    #    'Pipeline([("univariate_select", SelectPercentile(f_classif, 33)), ("L1_select", RandomizedLasso(random_state=42, n_jobs=self.cvJobs))])
    #featureSelectionString = 'Pipeline([("1_univariate_select", SelectFwe(f_classif, alpha=30.0)), ("2_rpca", RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/max(1.5,len(self.featureGetters)))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])'


    #dimensionality reduction (TODO: make a separate step than feature selection)
    #featureSelectionString = 'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/max(1.5,len(self.featureGetters)))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)' ### TRY THIS ###
    #featureSelectionString = 'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/len(self.featureGetters))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)'#smaller among 10% or number of rows / number of feature tables
    #featureSelectionString = 'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/2)), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)'#smaller among 10% or number of rows / 2
    # featureSelectionString = 'RandomizedPCA(n_components=min(X.shape[1], int(X.shape[0]/4)), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=min(X.shape[1], 2000), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=int(X.shape[1]*.10), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'RandomizedPCA(n_components=min(int(X.shape[0]*1.5), X.shape[1]), random_state=42, whiten=False, iterated_power=3)'
    #featureSelectionString = 'PCA(n_components=min(int(X.shape[1]*.10), X.shape[0]), whiten=False)'
    #featureSelectionString = 'PCA(n_components=0.99, whiten=False)'
    #featureSelectionString = 'VERPCA(n_components=0.999, whiten=False, max_components_ratio = min(1, X.shape[0]/float(X.shape[1])))'
    #featureSelectionString = 'KernelPCA(n_components=int(X.shape[1]*.02), kernel="rbf", degree=3, eigen_solver="auto")'  


    featureSelectMin = 50 #must have at least this many features to perform feature selection
    featureSelectPerc = 1.00 #only perform feature selection on a sample of training (set to 1 to perform on all)
    #featureSelectPerc = 0.20 #only perform feature selection on a sample of training (set to 1 to perform on all)

    testPerc = .20 #percentage of sample to use as test set (the rest is training)
    randomState = 42 #percentage of sample to use as test set (the rest is training)
    #randomState = 64 #percentage of sample to use as test set (the rest is training)

    trainingSize = 1000000 #if this is smaller than the training set, then it will be reduced to this. 

    def __init__(self, og, fgs, modelName = 'svc'):
        #initialize classification predictor
        self.outcomeGetter = og

        #setup feature getters:

        if not isinstance(fgs, collections.Iterable):
            fgs = [fgs]
        self.featureGetters = fgs
        self.featureGetter = fgs[0] #legacy support

        #setup other params / instance vars
        self.modelName = modelName
        self.classificationModels = dict()
        self.scalers = dict()
        self.fSelectors = dict()
        self.featureNames = [] #holds the order the features are expected in
        self.multiFSelectors = None
        self.multiScalers = None
        self.multiXOn = False #whether multiX was used for training

    def train(self, standardize = True, sparse = False, restrictToGroups = None, groupsWhere = ''):
        """Tests classifier, by pulling out random testPerc percentage as a test set"""
        
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
            for controlName, controlValues in controls.iteritems():
                controls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % (len(groups))

        ####################
        #2. get data for Xs:
        (groupNormsList, featureNamesList) = ([], [])
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        UGroups = None
        for fg in self.featureGetters:
            (groupNorms, featureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            groupNormValues = [groupNorms[feat] for feat in featureNames] #list of dictionaries of group_id => group_norm
            groupNormsList.append(groupNormValues)
            #print featureNames[:10]#debug
            featureNamesList.append(featureNames)
            fgGroups = getGroupsFromGroupNormValues(groupNormValues)
            if not XGroups:
                XGroups = set(fgGroups)
                UGroups = set(fgGroups)
            else:
                XGroups = XGroups & fgGroups #intersect groups from all feature tables
                UGroups = UGroups | fgGroups #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should

        XGroups = XGroups | groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        UGroups = UGroups | groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        
        ################################
        #2b) setup control data:
        controlValues = allControls.values()
        if controlValues:
            groupNormsList.append(controlValues)

        #########################################
        #3. train for all possible ys:
        self.multiXOn = True
        (self.classificationModels, self.multiScalers, self.multiFSelectors) = (dict(), dict(), dict())
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            multiXtrain = list()
            #trainGroupsOrder = list(XGroups & set(outcomes.keys()))
            trainGroupsOrder = list(UGroups & set(outcomes.keys()))
            for i in xrange(len(groupNormsList)):
                groupNormValues = groupNormsList[i]
                #featureNames = featureNameList[i] #(if needed later, make sure to add controls to this)
                (Xdicts, ydict) = (groupNormValues, outcomes)
                print "  (feature group: %d)" % (i)
                (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                if len(ytrain) > self.trainingSize:
                    Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                multiXtrain.append(Xtrain)
                print "   [Train size: %d ]" % (len(ytrain))

            #############
            #4) fit model
            (self.classificationModels[outcomeName], self.multiScalers[outcomeName], self.multiFSelectors[outcomeName]) = \
                self._multiXtrain(multiXtrain, ytrain, standardize, sparse = sparse)

        print "\n[TRAINING COMPLETE]\n"
        self.featureNamesList = featureNamesList

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
            print "\n!!! WARNING: multiple feature tables passed in rp.test only handles one for now.\n        try --combo_test_classification or --predict_classification !!!\n"
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

            (classifier, scaler, fSelector) = self._train(Xtrain, ytrain, standardize)
            ypred = self._predict(classifier, Xtest, scaler = scaler, fSelector = fSelector)
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
                (self.classificationModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName]) = (classifier, scaler, fSelector)

        print "\n[TEST COMPLETE]\n"

    #####################################################
    ######## Main Testing Method ########################
    def testControlCombos(self, standardize = True, sparse = False, saveModels = False, 
                          blacklist = None, noLang = False, allControlsOnly = False, comboSizes = None, 
                          nFolds = 2, savePredictions = False, weightedEvalOutcome = None, 
                          stratifyFolds = True, warmStartControls = False, controlCombineProbs = True, groupsWhere = '',
                          adaptTables = None, adaptColumns = None):
        """Tests classifier, by pulling out random testPerc percentage as a test set"""

        ##TODO ADD RESIDUALIZED CONTROLS: use "warm_start" with ETC or gradient boosting
        
        ###################################
        #1. setup groups for random folds
        if blacklist: print "USING BLACKLIST: %s" %str(blacklist)
        (groups, allOutcomes, allControls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        groupFolds = []
        if not stratifyFolds: 
            print "[number of groups: %d (%d Folds)] non-stratified / using same folds for all outcomes" % (len(groups), nFolds)
            random.seed(self.randomState)
            groupList = list(groups)
            random.shuffle(groupList)
            groupFolds = foldN(groupList, nFolds)
        else:
            print "    using stratified folds; different folds per outcome"

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
        XGroups = None #(intersect) holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        UGroups = None #(union) holds the set of groups from any feature table
        for fg in self.featureGetters:
            #(groupNorms, featureNames) = (None, None)
            if blacklist:
                print "!!!WARNING: USING BLACKLIST WITH SPARSE IS NOT CURRENTLY SUPPORTED!!!"
            (groupNorms, featureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            groupNormValues = groupNorms.values() #list of dictionaries of group_id => group_norm
            groupNormsList.append(groupNormValues)
            featureNamesList.append(featureNames)
            fgGroups = getGroupsFromGroupNormValues(groupNormValues)
            if not XGroups:
                XGroups = set(fgGroups)
                UGroups = set(fgGroups)
            else:
                XGroups = XGroups & fgGroups #intersect groups from all feature tables
                UGroups = UGroups | fgGroups #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should
        XGroups = XGroups & groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        UGroups = UGroups & groups
        
        ################################
        #2b) setup control combinations:
        controlKeys = allControls.keys()
        scores = dict() #outcome => control_tuple => [0],[1] => scores= {R2, R, r, r-p, rho, rho-p, MSE, train_size, test_size, num_features,yhats}
        if not comboSizes:
            comboSizes = xrange(len(controlKeys)+1)
            if allControlsOnly:
                #if len(controlKeys) > 1: 
                #    comboSizes = [0, 1, len(controlKeys)]
                #else:
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
                    #thisOutcomeGroups = set(outcomes.keys()) & XGroups #this intersects with XGroups maybe use UGroups?
                    thisOutcomeGroups = set(outcomes.keys()) & UGroups #this intersects with UGroups
                    if not outcomeName in scores:
                        scores[outcomeName] = dict()
                        
                    if stratifyFolds: 
                        #even classes across the outcomes
                        print "Warning: Stratifying outcome classes across folds (thus, folds will differ across outcomes)."
                        groupFolds = stratifyGroups(thisOutcomeGroups, outcomes, nFolds, randomState = 42)

                    #for warmStartControls or controlCombinedProbs
                    lastClassifiers= [None]*nFolds
                    savedControlYpp = None

                    for withLanguage in xrange(2):
                        classifier = None
                        if withLanguage: 
                            if noLang or (allControlsOnly and (r > 1) and (r < len(controlKeys))):#skip to next
                                continue
                            print "\n= %s (w/ lang.)=\n%s"%(outcomeName, '-'*(len(outcomeName)+14))
                        elif controlValues: 
                            print "\n= %s (NO lang.)=\n%s"%(outcomeName, '-'*(len(outcomeName)+14))
                        else: #no controls in this iteration
                            continue

                        testStats = {'acc': [], 'f1': [], 'precision': [], 'rho': [], 'rho-p': [], 'recall': [], 'mfclass_acc': [], 'auc': []}
                        if wOutcome:
                            testStats.update({'r-wghtd' : [], 'r-wghtd-p' : []})
                        predictions = {}
                        predictionProbs = {}


                        ###############################
                        #3a) iterate over nfold groups:
                        for testChunk in xrange(0, len(groupFolds)):
                            trainGroups = set()
                            for chunk in (groupFolds[:testChunk]+groupFolds[(testChunk+1):]):
                                trainGroups.update(chunk)
                            testGroups = set(groupFolds[testChunk])
                            #set static group order across features:
                            trainGroupsOrder = list(thisOutcomeGroups & trainGroups)
                            testGroupsOrder = list(thisOutcomeGroups & testGroups)
                            testSize = len(testGroupsOrder)
                            print "Fold %d " % (testChunk)

                            ###########################################################################
                            #3b)setup train and test data (different X for each set of groupNormValues)
                            (multiXtrain, multiXtest, ytrain, ytest) = ([], [], None, None) #ytrain, ytest should be same across tables
                            #get the group order across all
                            gnListIndices = range(len(thisGroupNormsList))
                            num_feats = 0;
                            if not withLanguage:
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
                            mfclass = Counter(ytrain).most_common(1)[0][0]
                            (classifier, multiScalers, multiFSelectors) = (None, None, None)
                            if warmStartControls:
                                (classifier, multiScalers, multiFSelectors) = self._multiXtrain(multiXtrain, ytrain, standardize, sparse = sparse, warmStartClassifier = lastClassifiers[testChunk])
                                lastClassifiers[testChunk] = classifier
                            else:
                                (classifier, multiScalers, multiFSelectors) = self._multiXtrain(multiXtrain, ytrain, standardize, sparse = sparse)
                                

                            ypredProbs, ypredClasses = self._multiXpredict(classifier, multiXtest, \
                                                                           multiScalers = multiScalers, multiFSelectors = multiFSelectors, sparse = sparse, probs=True)
                            ypred = ypredClasses[ypredProbs.argmax(axis=1)]
                            predictions.update(dict(izip(testGroupsOrder,ypred)))
                            predictionProbs.update(dict(izip(testGroupsOrder,ypredProbs)))

                            acc = accuracy_score(ytest, ypred)
                            f1 = f1_score(ytest, ypred)
                            auc = pos_neg_auc(ytest, ypredProbs[:,-1])
                            # classes = list(set(ytest))
                            # ytest_binary = label_binarize(ytest,classes=classes)
                            # ypred_binary = label_binarize(ypred,classes=classes)
                            # auc = roc_auc_score(ytest_binary, ypred_binary)
                            testCounter = Counter(ytest)
                            mfclass_acc = testCounter[mfclass] / float(len(ytest))
                            print " *confusion matrix: \n%s"% str(confusion_matrix(ytest, ypred))
                            print " *precision and recall: \n%s" % classification_report(ytest, ypred)
                            print " *FOLD ACC: %.4f (mfclass_acc: %.4f); mfclass: %s; auc: %.4f\n" % (acc, mfclass_acc, str(mfclass), auc)
                            testStats['acc'].append(acc)
                            testStats['f1'].append(f1)
                            testStats['auc'].append(auc)
                            (rho, rho_p) = spearmanr(ytest, ypred)
                            testStats['rho'].append(rho)
                            testStats['rho-p'].append(rho_p)
                            testStats['precision'].append(precision_score(ytest, ypred))
                            testStats['recall'].append(recall_score(ytest, ypred))
                            testStats['mfclass_acc'].append(mfclass_acc)
                            testStats.update({'train_size': len(ytrain), 'test_size': len(ytest), 'num_features' : num_feats, 
                             '{model_desc}': str(classifier).replace('\t', "  ").replace('\n', " ").replace('  ', " "),
                             '{modelFS_desc}': str(multiFSelectors[0]).replace('\t', "  ").replace('\n', " ").replace('  ', " "),
                             'mfclass' : str(mfclass), 'num_classes' : str(len(testCounter.keys()))
                                              })
                            ##4 b) weighted eval
                            if wOutcome:
                                weights = [float(wOutcome[k]) for k in testGroupsOrder]
                                try:
                                    results = sm.WLS(zscore(ytest), zscore(ypred), weights).fit() #runs classification
                                    testStats['r-wghtd'].append(results.params[-1])
                                    testStats['r-wghtd-p'].append(results.pvalues[-1])
                                    #print results.summary(outcomeName, [outcomeName+'_pred'])#debug
                                except ValueError as err:
                                    print "WLS threw ValueError: %s\nresult not included" % str(err)

                        #########################
                        #5) aggregate test stats:
                        reportStats = dict()
                        for k, v in testStats.items():
                            if isinstance(v, list):
                                reportStats['folds_'+k] = mean(v)
                                reportStats['folds_se_'+k] = std(v) / sqrt(float(nFolds))
                            else:
                                reportStats[k] = v
                                
                        #overall stats:
                        ytrue, ypred, ypredProbs = alignDictsAsy(outcomes, predictions, predictionProbs)
                        ypredProbs = array(ypredProbs)
                        reportStats['acc'] = accuracy_score(ytrue, ypred)
                        reportStats['f1'] = f1_score(ytrue, ypred)
                        reportStats['a'] = pos_neg_auc(ytrue, ypredProbs[:,-1])

                        reportStats['auc_cntl_comb'] = 0.0
                        reportStats['auc_cntl_comb2'] = 0.0
                        reportStats['auc_cntl_comb_p'] = 1.0
                        reportStats['auc_cntl_comb_t'] = 0.0
                        if controlCombineProbs:
                            if withLanguage and savedControlYpp is not None:
                                newprobs = np.array(easyNFoldLR( np.array([ypredProbs[:,-1], savedControlYpp]).T, ytrue, len(groupFolds)))[:,-1]
                                newprobs2 = np.array(easyNFoldAUCWeight( np.array([ypredProbs[:,-1], savedControlYpp]).T, ytrue, len(groupFolds)))[:,-1]
                                #newprobs2 = (ypredProbs[:,-1] + savedControlYpp)/2.0
                                #newprobs = ensembleDecisionFuncs(outcomes, groupFolds, ypredProbs[:,-1], savedControlYpp)
                                reportStats['auc_cntl_comb'] = pos_neg_auc(ytrue, newprobs)
                                reportStats['auc_cntl_comb_t'], reportStats['auc_cntl_comb_p'] = paired_t_1tail_on_errors(newprobs, savedControlYpp, ytrue)
                                reportStats['auc_cntl_comb2'] = pos_neg_auc(ytrue, newprobs2)
                            else: #save probs for next round
                                savedControlYpp = ypredProbs[:,-1]

                        testCounter = Counter(ytrue)
                        reportStats['mfclass_acc'] = testCounter[mfclass] / float(len(ytrue))
                        if savePredictions: 
                            reportStats['predictions'] = predictions
                            reportStats['predictionProbs'] = {k:v[-1] for k,v in predictionProbs.iteritems()}
                        #pprint(reportStats) #debug
                        mfclass = Counter(ytest).most_common(1)[0][0]
                        print "* Overall Fold Acc: %.4f (+- %.4f) vs. MFC Accuracy: %.4f (based on test rather than train)"% (reportStats['folds_acc'], reportStats['folds_se_acc'], reportStats['folds_mfclass_acc'])
                        print "*       Overall F1: %.4f (+- %.4f)"% (reportStats['folds_f1'], reportStats['folds_se_f1'])
                        print "       + precision: %.4f (+- %.4f)"% (reportStats['folds_precision'], reportStats['folds_se_precision'])
                        print "       +    recall: %.4f (+- %.4f)"% (reportStats['folds_recall'], reportStats['folds_se_recall'])
                        if saveModels: 
                            print "!!SAVING MODELS NOT IMPLEMENTED FOR testControlCombos!!"
                        try:
                            scores[outcomeName][controlKeyCombo][withLanguage] = reportStats
                        except KeyError:
                            scores[outcomeName][controlKeyCombo] = {withLanguage: reportStats}


        print "\n[TEST COMPLETE]\n"
        return scores
    #################################################
    #################################################

    def predict(self, standardize = True, sparse = False, restrictToGroups = None, groupsWhere = ''):
        
        if not self.multiXOn:
            print "model trained without multiX, reverting to old predict"
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
                controls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % (len(groups))

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
            #print self.featureNamesList[i][:10]#debug
            for feat in self.featureNamesList[i]:
                if feat in groupNorms:
                    groupNormValues.append(groupNorms[feat])
                else:
                    groupNormValues.append(dict())
            groupNormsList.append(groupNormValues)
            print "  Features Aligned: %d" % len(groupNormValues)
            fgGroups = getGroupsFromGroupNormValues(groupNormValues)
            if not XGroups:
                XGroups = set(fgGroups)
                UGroups = set(fgGroups)
            else:
                XGroups = XGroups & fgGroups #intersect groups from all feature tables
                UGroups = UGroups | fgGroups #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should
                #potential source of bug: if a sparse feature table doesn't have all of the groups which it should
        #XGroups = XGroups & groups #this should not be needed
        if len(XGroups) < len(groups): 
            print " Different number of groups available for different outcomes."
        
        ################################
        #2b) setup control data:
        controlValues = allControls.values()
        if controlValues:
            groupNormsList.append(controlValues)

        #########################################
        #3. predict for all possible outcomes
        predictions = dict()
        testGroupsOrder = list(UGroups) 
        #testGroupsOrder = list(XGroups) 
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
            ypred = self._multiXpredict(self.classificationModels[outcomeName], multiXtest, multiScalers = self.multiScalers[outcomeName], \
                                            multiFSelectors = self.multiFSelectors[outcomeName], sparse = sparse)
            print "[Done. Evaluation:]"
            acc = accuracy_score(ytest, ypred)
            f1 = f1_score(ytest, ypred)
            testCounter = Counter(ytest)
            mfclass = Counter(ytest).most_common(1)[0][0]
            mfclass_acc = testCounter[mfclass] / float(len(ytest))
            print " *confusion matrix: \n%s"% str(confusion_matrix(ytest, ypred))
            print " *precision and recall: \n%s" % classification_report(ytest, ypred)
            print " *ACC: %.4f (mfclass_acc: %.4f); mfclass: %s\n" % (acc, mfclass_acc, str(mfclass))

            mse = metrics.mean_squared_error(ytest, ypred)
            print "*Mean Squared Error:                 %.4f"% mse
            assert len(thisTestGroupsOrder) == len(ypred), "can't line predictions up with groups" 
            predictions[outcomeName] = dict(zip(thisTestGroupsOrder,ypred))

        print "[Prediction Complete]"

        return predictions

    def predictNoOutcomeGetter(self, groups, standardize = True, sparse = False, restrictToGroups = None):
        
        outcomes = list(self.classificationModels.keys())

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
                #potential source of bug: if a sparse feature table doesn't have all of the groups which it should
        #XGroups = XGroups & groups #this should not be needed
        if len(XGroups) < len(groups): 
            print " Different number of groups available for different outcomes. (%d, %d)" % (len(XGroups), len(groups))
        

        #########################################
        #3. predict for all possible outcomes
        predictions = dict()
        testGroupsOrder = list(UGroups) 
        #testGroupsOrder = list(XGroups)
        
        for outcomeName in sorted(outcomes):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            thisTestGroupsOrder = testGroupsOrder

            multiXtest = []
            for i in xrange(len(groupNormsList)): #get each feature group data (i.e. feature table)
                
                groupNormValues = groupNormsList[i] # basically a feature table in list(dict) form
                # We want the dictionaries to turn into a list that is aligned
                gns = dict(zip(self.featureNamesList[i], groupNormValues))

                df = pd.DataFrame(data=gns)
                df = df.fillna(0.0)
                df = df[self.featureNamesList[i]]
                df = df.reindex(thisTestGroupsOrder)
                print "  (feature group: %d)" % (i)

                multiXtest.append(csr_matrix(df.values))
                
            #############
            #4) predict
            ypred = self._multiXpredict(self.classificationModels[outcomeName], multiXtest, multiScalers = self.multiScalers[outcomeName], \
                                            multiFSelectors = self.multiFSelectors[outcomeName], sparse = sparse)
            print "[Done.]"

            assert len(thisTestGroupsOrder) == len(ypred), "can't line predictions up with groups" 
            predictions[outcomeName] = dict(zip(thisTestGroupsOrder,ypred))

        print "[Prediction Complete]"

        return predictions


    def predictAllToFeatureTable(self, standardize = True, sparse = False, fe = None, name = None, nFolds = 10, groupsWhere = ''):
        if not fe:
            print "Must provide a feature extractor object"
            sys.exit(0)

        #1. get all groups
        
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        #split groups into 10 chunks
        groups = list(groups)
        random.seed(self.randomState)
        random.shuffle(groups)        
        chunks = foldN(groups, nFolds)
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
        featureTableName = fe.createFeatureTable(featureName, "VARCHAR(%d)"%featLength, 'INTEGER')

        for feat in featNames:
            preds = predictions[feat]

            print "[Inserting Predictions as Feature values for %s]" % feat
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, '"""+feat+"""', %s, %s)"""
            rows = [(k, v, v) for k, v in preds.iteritems()] #adds group_norm and applies freq filter
            mm.executeWriteMany(fe.corpdb, fe.dbCursor, wsql, rows, writeCursor=fe.dbConn.cursor(), charset=fe.encoding)

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
            chunks =  foldN(groups, numChunks)

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

    def predictToFeatureTable(self, standardize = True, sparse = False, fe = None, name = None, groupsWhere = ''):
        if not fe:
            print "Must provide a feature extractor object"
            sys.exit(0)

        #handle large amount of predictions:
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(groupsWhere = groupsWhere)
        groups = list(groups)
        chunks = [groups]
        #split groups into chunks
        if len(groups) > self.maxPredictAtTime:
            numChunks = int(len(groups) / float(self.maxPredictAtTime)) + 1
            print "TOTAL GROUPS (%d) TOO LARGE. Breaking up to run in %d chunks." % (len(groups), numChunks)
            random.seed(self.randomState)
            random.shuffle(groups)        
            chunks =  foldN(groups, numChunks)
        predictions = dict()
        totalPred = 0
        c = 0
        for chunk in chunks:
            print "\n\n**CHUNK %d\n" % c
            if len(chunk) == len(groups):
                chunk = None
            chunkPredictions = self.predict(standardize, sparse, chunk)
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
            c+=1

        featNames = predictions.keys()
        featLength = max(map(lambda s: len(s), featNames))

        #CREATE TABLE:
        featureName = "p_%s" % self.modelName[:4]
        if name: featureName += '_' + name
        featureTableName = fe.createFeatureTable(featureName, "VARCHAR(%d)"%featLength, 'INTEGER')

        for feat in featNames:
            preds = predictions[feat]

            print "[Inserting %d Predictions as Feature values for %s]" % (len(preds), feat)
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, '"""+feat+"""', %s, %s)"""
            rows = []
            written = 0
            for k, v in preds.iteritems():
                rows.append((k, v, v))

            mm.executeWriteMany(fe.corpdb, fe.dbCursor, wsql, rows, writeCursor=fe.dbConn.cursor(), charset=fe.encoding)
            written += len(rows)
            print "   %d feature rows written" % written


    def getWeightsForFeaturesAsADict(self): 
        #returns dictionary of featureTable -> outcomes -> feature_name -> weight
        #weights: self.classificationModels[outcome].coef_
        # Scaler: self.multiScalers[outcome]

        weights_dict = dict()

        featTables = [fg.featureTable for fg in self.featureGetters]
        
        startIndex = 0 # for multiX classification
        for i, featTableFeats in enumerate(self.featureNamesList):
            weights_dict[featTables[i]] = dict()
            for outcome, model in self.classificationModels.iteritems():
                weights_dict[featTables[i]][outcome] = dict()
                coefficients = eval('self.classificationModels[outcome].%s' % self.modelToCoeffsName[self.modelName.lower()])
                #pprint(coefficients) #debug
                
                intercept = self.classificationModels[outcome].intercept_ if 'intercept_' in dir(self.classificationModels[outcome]) else None
                if isinstance(intercept, np.ndarray) or isinstance(intercept, list):
                    if len(intercept) == 1:
                        intercept = intercept[0]
                    else:
                        raise ValueError("Intercept is not an array of lenght 1 or a float...")
                    
                print "Intercept[%s]: %10.8f" % (outcome, intercept)
                #print dir( self.classificationModels[outcome])
                #print dir(self.multiFSelectors[outcome][i]) # debug
                
                # Getting the right chunk if there are multiple feature tables (OLD)
                # coefficients = coefficients[startIndex:(startIndex+len(featTableFeats))]
                
                if coefficients.shape[0] != 1:
                    coefficients.resize(1,len(coefficients))
                # Inverting Feature Selection
                
                if self.multiFSelectors[outcome][i]:
                    print "Inverting the feature selection: %s" % self.multiFSelectors[outcome][i]
                    coefficients = self.multiFSelectors[outcome][i].inverse_transform(coefficients)#.flatten()

                    if 'mean_' in dir(self.multiFSelectors[outcome][i]): # PCA, RPCA
                        print "RPCA mean: ", self.multiFSelectors[outcome][i].mean_

                    if 'steps' in dir(self.multiFSelectors[outcome][i]): # Pipeline(1_UFS, 2_PCA)
                        if 'mean_' in dir(self.multiFSelectors[outcome][i].steps[1][1]):
                            print self.multiFSelectors[outcome][i].steps[1][1].mean_, len(self.multiFSelectors[outcome][i].steps[1][1].mean_)
                            mean = self.multiFSelectors[outcome][i].steps[1][1].mean_.copy()
                            t = self.multiFSelectors[outcome][i].steps[0][1].inverse_transform(mean).flatten()
                            print self.multiFSelectors[outcome][i].steps[1][1].transform(mean-intercept), self.multiFSelectors[outcome][i].steps[1][1].transform(mean-intercept).sum()
                            print t, len(t)
                            print coefficients
                            # coefficients = coefficients + t
                            print coefficients
                            print "Pipelines don't work with this option (predtolex)"
                            sys.exit()


                # coefficients.resize(1,len(coefficients))
                # print coefficients.shape
                # Inverting the scaling
                if self.multiScalers[outcome][i]:
                    print "Standardscaler doesn't work with Prediction To Lexicon"
                    # print self.multiScalers[outcome][i].mean_, self.multiScalers[outcome][i].std_
                    # coefficients = self.multiScalers[outcome][i].inverse_transform(coefficients).flatten()
                    coefficients = coefficients.flatten()
                else:
                    coefficients = coefficients.flatten()
                
                # featTableFeats contains the list of features
                if len(coefficients) != len(featTableFeats):
                    print "length of coefficients (%d) does not match number of features (%d)" % (len(coefficients), len(featTableFeats))
                    sys.exit(1)

                weights_dict[featTables[i]][outcome] = {featTableFeats[j]: coefficients[j] for j in xrange(len(featTableFeats))}
                weights_dict[featTables[i]][outcome]['_intercept'] = intercept

            startIndex += len(featTableFeats)
        #pprint(weights_dict)
        return weights_dict


    def _train(self, X, y, standardize = True):
        """does the actual classification training, first feature selection: can be used by both train and test"""

        sparse = True
        if not isinstance(X, csr_matrix):
            X = np.array(X)
            sparse = False
       
        scaler = None
        #print " Standardize: ", standardize
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
                              self.cvParams[modelName], n_jobs = self.cvJobs,
                              cv=ShuffleSplit(len(y), n_iterations=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))
            print "[Performing grid search for parameters over training]"
            gs.fit(X, y)

            print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
            return gs.best_estimator_, scaler, fSelector
        else:
            # no grid search
            print "[Training classification model: %s]" % modelName
            classifier = eval(self.modelToClassName[modelName]+'()')
            classifier.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
            #print dict(self.cvParams[modelName][0])

            classifier.fit(X, y)
            #print "coefs"
            #print classifier.coef_
            print "model: %s " % str(classifier)
            if modelName[-2:] == 'cv' and 'alphas' in classifier.get_params():
                print "  selected alpha: %f" % classifier.alpha_
            return classifier, scaler, fSelector


    def _multiXtrain(self, X, y, standardize = True, sparse = False, warmStartClassifier = None):
        """does the actual classification training, first feature selection: can be used by both train and test
           create multiple scalers and feature selectors
           and just one classification model (of combining the Xes into 1)
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
            #print " Standardize: ", standardize
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
            try:
                X = matrixAppendHoriz(X, nextX)
            except ValueError as e:
                print "ValueError: %s" % str(e)
                print "couldn't append arrays: perhaps one is sparse and one dense"
                sys.exit(0)
        modelName = self.modelName.lower()
        if (X.shape[1] / float(X.shape[0])) < self.backOffPerc: #backoff to simpler model:
            print "number of features is small enough, backing off to %s" % self.backOffModel
            modelName = self.backOffModel.lower()

        if hasMultValuesPerItem(self.cvParams[modelName]) and modelName[-2:] != 'cv':
            #grid search for classifier params:
            if warmStartClassifier:
                raise Exception('Warm start classifier not implemented with grid search')
            gs = GridSearchCV(eval(self.modelToClassName[modelName]+'()'), 
                              self.cvParams[modelName], n_jobs = self.cvJobs,
                              cv=ShuffleSplit(len(y), n_iter=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))
            print "[Performing grid search for parameters over training]"
            gs.fit(X, y)

            print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
            return gs.best_estimator_,  multiScalers, multiFSelectors
        else:
            # no grid search
            print "[Training classification model: %s]" % modelName
            classifier = None
            if warmStartClassifier:
                classifier = warmStartClassifier
                classifier.set_params(warm_start=True)
            if not classifier:
                classifier = eval(self.modelToClassName[modelName]+'()')
            try: 
                classifier.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
            except IndexError: 
                print "No CV parameters available"
                raise IndexError
            #print dict(self.cvParams[modelName][0])

            classifier.fit(X, y)
            #print "coefs"
            #print classifier.coef_
            print "model: %s " % str(classifier)
            if modelName[-2:] == 'cv' and 'alphas' in classifier.get_params():
                print "  selected alpha: %f" % classifier.alpha_
            return classifier, multiScalers, multiFSelectors

    def _predict(self, classifier, X, scaler = None, fSelector = None, y = None):
        if scaler:
            X = scaler.transform(X)
        if fSelector:
            newX = fSelector.transform(X)
            if newX.shape[1]:
                X = newX
            else:
                print "No features selected, so using original full X"

        return classifier.predict(X)

    def roc(self, standardize = True, sparse = False, restrictToGroups = None, output_name = None, groupsWhere = ''):
        """Tests classifier, by pulling out random testPerc percentage as a test set"""
        
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
            for controlName, controlValues in controls.iteritems():
                controls[controlName] = dict([(g, controlValues[g]) for g in groups])
        print "[number of groups: %d]" % (len(groups))

        ####################
        #2. get data for Xs:
        (groupNormsList, featureNamesList) = ([], [])
        XGroups = None #holds the set of X groups across all feature spaces and folds (intersect with this to get consistent keys across everything
        UGroups = None
        for fg in self.featureGetters:
            (groupNorms, featureNames) = fg.getGroupNormsSparseFeatsFirst(groups)
            groupNormValues = [groupNorms[feat] for feat in featureNames] #list of dictionaries of group_id => group_norm
            groupNormsList.append(groupNormValues)
            #print featureNames[:10]#debug
            featureNamesList.append(featureNames)
            fgGroups = getGroupsFromGroupNormValues(groupNormValues)
            if not XGroups:
                XGroups = set(fgGroups)
                UGroups = set(fgGroups)
            else:
                XGroups = XGroups & fgGroups #intersect groups from all feature tables
                UGroups = UGroups | fgGroups #intersect groups from all feature tables
                #potential source of bug: if a sparse feature table doesn't have all the groups it should

        XGroups = XGroups | groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        UGroups = UGroups | groups #probably unnecessary since groups was given when grabbing features (in which case one could just use groups)
        
        ################################
        #2b) setup control data:
        controlValues = allControls.values()
        if controlValues:
            groupNormsList.append(controlValues)

        #########################################
        #3. train for all possible ys:
        self.multiXOn = True
        (self.classificationModels, self.multiScalers, self.multiFSelectors) = (dict(), dict(), dict())
        for outcomeName, outcomes in sorted(allOutcomes.items()):
            print "\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4))
            multiXtrain = list()
            #trainGroupsOrder = list(XGroups & set(outcomes.keys()))
            trainGroupsOrder = list(UGroups & set(outcomes.keys()))
            for i in xrange(len(groupNormsList)):
                groupNormValues = groupNormsList[i]
                #featureNames = featureNameList[i] #(if needed later, make sure to add controls to this)
                (Xdicts, ydict) = (groupNormValues, outcomes)
                print "  (feature group: %d)" % (i)
                (Xtrain, ytrain) = alignDictsAsXy(Xdicts, ydict, sparse=True, keys = trainGroupsOrder)
                if len(ytrain) > self.trainingSize:
                    Xtrain, Xthrowaway, ytrain, ythrowaway = train_test_split(Xtrain, ytrain, test_size=len(ytrain) - self.trainingSize, random_state=self.randomState)
                multiXtrain.append(Xtrain)
                print "   [Train size: %d ]" % (len(ytrain))

            #############
            #4) fit model
            (self.classificationModels[outcomeName], self.multiScalers[outcomeName], self.multiFSelectors[outcomeName]) = \
                self._roc(multiXtrain, ytrain, standardize, sparse = sparse, output_name = output_name)

        print "\n[TRAINING COMPLETE]\n"
        self.featureNamesList = featureNamesList

    def _roc(self, X, y, standardize = True, sparse = False, output_name = None):

        X = X
        y = label_binarize(y, classes=list(set(y)))

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
            #print " Standardize: ", standardize
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
            try:
                X = matrixAppendHoriz(X, nextX)
            except ValueError as e:
                print "ValueError: %s" % str(e)
                print "couldn't append arrays: perhaps one is sparse and one dense"
                sys.exit(0)
        modelName = self.modelName.lower()
        if (X.shape[1] / float(X.shape[0])) < self.backOffPerc: #backoff to simpler model:
            print "number of features is small enough, backing off to %s" % self.backOffModel
            modelName = self.backOffModel.lower()

        # Here is the place to do ROC
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
        y_score = y_test
        
        ret = None
        if hasMultValuesPerItem(self.cvParams[modelName]) and modelName[-2:] != 'cv':
            # multiple Cs or alphas supplied
            # Split into all cases for all classes of y

            for i in xrange(y_train.shape[1]):
                y_train_slice = y_train[:,i]
                y_test_slice = y_test[:,i]
                # grid search for classifier params:
                gs = GridSearchCV(eval(self.modelToClassName[modelName]+'()'), 
                                  self.cvParams[modelName], n_jobs = self.cvJobs,
                                  cv=ShuffleSplit(len(y_train_slice), n_iter=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))
                print "[Performing grid search for parameters over training for class: %d]" % i
                gs.fit(X_train, y_train_slice)
                fitted_model = gs.best_estimator_
                y_score_slice = fitted_model.decision_function(X_test)
                y_score[:,i] = y_score_slice
                print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
            ret = (gs.best_estimator_,  multiScalers, multiFSelectors)
            print y_score
        else:
            print "[ROC - Using classification model: %s]" % modelName
            classifier = eval(self.modelToClassName[modelName]+'()')
            classifier.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
            classifier = OneVsRestClassifier(classifier)
            print "[Training classifier]"
            fitted_model = classifier.fit(X_train, y_train)
            print "[Done training]"
            y_score = fitted_model.decision_function(X_test)
            ret = (fitted_model,  multiScalers, multiFSelectors)

        self.roc_curves(y_test, y_score, output_name)
        
        return ret
    """
    # no grid search
    print "[Training classification model: %s]" % modelName
    classifier = eval(self.modelToClassName[modelName]+'()')
    try: 
    classifier.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in self.cvParams[modelName][0].iteritems()))
    except IndexError: 
    print "No CV parameters available"
    raise IndexError
    #print dict(self.cvParams[modelName][0])
    
    classifier.fit(X, y)
    #print "coefs"
    #print classifier.coef_
    print "model: %s " % str(classifier)
    if modelName[-2:] == 'cv' and 'alphas' in classifier.get_params():
    print "  selected alpha: %f" % classifier.alpha_"""

    def roc_curves(self, y_test, y_score, output_name = None):
        output_name = "ROC" if not output_name else output_name
        pp = PdfPages(output_name+'.pdf' if output_name[-4:] != '.pdf' else output_name)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_test.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        print "ROC AUC (area under the curve)", roc_auc
        
        # Plot ROC curve
        figure = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        pp.savefig(figure)
        
        pp.close()

    def _multiXpredict(self, classifier, X, multiScalers = None, multiFSelectors = None, y = None, sparse = False, probs = False):
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
                X = scaler.transform(X)
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
            X = matrixAppendHoriz(X, nextX)

        print "  predict: combined X shape: %s" % str(X.shape) #debug
        if hasattr(classifier, 'intercept_'):
            print "  predict: classifier intercept: %s" % str(classifier.intercept_)

        if probs:
            try:  
                return classifier.predict_proba(X), classifier.classes_
            except AttributeError:
                confs = classifier.decision_function(X)
                if len(classifier.classes_) == 2:
                    confs = array(zip([-1*c for c in confs], confs))
                #note: may need to convert to probabilities by dividing by max
                return confs, classifier.classes_
                
        else:
            return classifier.predict(X)
    
        # print "Predicting", time.strftime('%c')
        # ret = classifier.predict(X)
        # print "Done", time.strftime('%c')
        # return ret



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
                  'classificationModels': self.classificationModels,
                  'multiScalers': self.multiScalers,
                  'scalers' : self.scalers, 
                  'multiFSelectors': self.multiFSelectors,
                  'fSelectors' : self.fSelectors,
                  'featureNames' : self.featureNames,
                  'featureNamesList' : self.featureNamesList,
                  'multiXOn' : self.multiXOn
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
            scoreNames = sorted(outcomeScores[rowKeys[0]].itervalues().next().keys(), key=str.lower)
            columnNames = ['row_id', 'outcome', 'model_controls'] + scoreNames + ['w/ lang.'] + controlNames

            #csv:
            csvOut = csv.DictWriter(outputstream, fieldnames=columnNames, delimiter=delimiter)
            if set(columnNames) != set(previousColumnNames):
                firstRow = dict([(str(k), str(k)) for k in columnNames])
                csvOut.writerow(firstRow)
                previousColumnNames = columnNames
            for rk in rowKeys:
                for withLang, sc in outcomeScores[rk].iteritems():
                    i+=1
                    rowDict = {'row_id': i, 'outcome': outcomeName, 'model_controls': str(rk)+str(withLang)}
                    for cn in controlNames:
                        rowDict[cn] = 1 if cn in rk else 0
                    rowDict['w/ lang.'] = withLang
                    rowDict.update({(k,v) for (k,v) in sc.items() if ((k is not 'predictions') and k is not 'predictionProbs')})
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

    @staticmethod
    def printComboControlPredictionProbsToCSV(scores, outputstream, paramString = None, delimiter='|'):
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
                     predictionData[str(i)+'_'+outcomeName+'_'+mc] = s['predictionProbs']
                     for k,v in s['predictionProbs'].iteritems():
                         data[k].append(v)

         writer = csv.writer(outputstream)
         writer.writerow(columns)
         for k,v in data.iteritems():
            v.insert(0,k)
            writer.writerow(v)
        
    #################
    ## Deprecated:
    def old_train(self, standardize = True, sparse = False, restrictToGroups = None):
        """Trains classification models"""
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
            (self.classificationModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName]) = self._train(X, y, standardize)

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
            (classifier, scaler, fSelector) = (self.classificationModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName])
            ypred = None
            if self.chunkPredictions:
                ypred = np.array([]) #chunks:
                for subX, suby in chunks(X, ytest, self.maxPredictAtTime):
                    ysubpred = self._predict(classifier, subX, scaler = scaler, fSelector = fSelector)
                    ypred = np.append(ypred, np.array(ysubpred))
                    print "   num predicticed: %d" % len(ypred)
            else:
                ypred = self._predict(classifier, X, scaler = scaler, fSelector = fSelector)
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

def xFoldN(l, folds):
    """ Yield successive n-sized chunks from l."""
    n = len(l) / folds
    last = len(l) % folds
    i = 0
    for f in xrange(folds):
        if last > 0:
            yield l[i:i+n+1]
            last -= 1
            i+= n+1
        else: 
            yield l[i:i+n]
            i+= n

def foldN(l, folds):
    return [f for f in xFoldN(l, folds)]

def stratifyGroups(groups, outcomes, folds, randomState = 42):
    """breaks groups up into folds such that each fold has at most 1 more of a class than other folds """
    random.seed(randomState)
    xGroups = sorted(list(set(groups) & set(outcomes.keys())))
    outcomesByClass = {}
    for g in xGroups:
        try: 
            outcomesByClass[outcomes[g]].append(g)
        except:
            outcomesByClass[outcomes[g]] = [g]

    groupsPerFold = {f: [] for f in xrange(folds)}
    countPerFold = {f: 0 for f in xrange(folds)}

    #add groups to folds per class, keeping track of balance
    for c, gs in outcomesByClass.iteritems():
        foldsOrder = [x[0] for x in sorted(countPerFold.items(), key=lambda x: (x[1], x[0]))]
        currentGFolds = foldN(gs, folds)
        i = 0
        for f in foldsOrder:
            groupsPerFold[f].extend(currentGFolds[i])
            countPerFold[f] += len(currentGFolds[i])
            i+=1

    #make sure all outcomes aren't together in the groups:
    for gs in groupsPerFold.values():                
        random.shuffle(gs)

    return groupsPerFold.values()

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

def matrixAppendHoriz(A, B):
    if isinstance(A, spmatrix) and isinstance(B, spmatrix):
        return hstack([A, B])
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.append(A, B, 1)
    elif isinstance(A, spmatrix) and isinstance(B, np.ndarray):
        return np.append(A.todense(), B, 1)
    elif isinstance(A, np.ndarray) and isinstance(B, npspmatrix):
        return np.append(A, B.todense(), 1)
    else:
        raise ValueError('matrix append types not supported: %s and %s' % (type(A).__name__, type(B).__name__))


#def mean(l):
#    return sum(l)/float(len(l))

def r2simple(ytrue, ypred):
    y_mean = sum(ytrue)/float(len(ytrue))
    ss_tot = sum((yi-y_mean)**2 for yi in ytrue)
    ss_err = sum((yi-fi)**2 for yi,fi in zip(ytrue,ypred))
    r2 = 1 - (ss_err/ss_tot)
    return r2

def easyNFoldLR(X, y, folds):
    """build logistic regressor given two sets of decision function outputs, return new probs """
    yscores = []
    X = zscore(X)
    y = np.array(y)
    print "[NFOLD TRAIN TESTING Easy LR]"
    groups = range(len(y))
    outcomes = dict(zip(groups, y))
    groupFolds = stratifyGroups(groups, outcomes, folds, randomState = 42)
    ymap = []
    for testgs in groupFolds:
        testgs = np.array(testgs)
        setgs = set(testgs)
        traings = [i for i in groups if i not in setgs]
        ytest = y[testgs]
        Xtest = X[testgs]
        ytrain = y[traings]
        Xtrain = X[traings]

        #lr = LogisticRegression(C = 100000, penalty = 'l2',  class_weight='auto')
        lr = LogisticRegression(C = 100000, penalty = 'l2',  class_weight='auto', fit_intercept=False)
        fitted_model = lr.fit(Xtrain, ytrain)
        #print fitted_model.coef_
        ypred_probs = fitted_model.predict_proba(Xtest)
        print "   EASY LR AUC: %.4f" %  pos_neg_auc(ytest, ypred_probs[:,-1])
        yscores.extend(ypred_probs)
        ymap.extend(testgs)
    newyscores = [0]*len(y)
    for i in groups:
        newyscores[ymap[i]] = yscores[i]
    print "[Done]"
    print " FINAL EASY LR AUC: %.4f" %  pos_neg_auc(y, np.array(newyscores)[:,-1])
    return newyscores

def easyNFoldAUCWeight(X, y, folds):
    """build logistic regressor given two sets of decision function outputs, return new probs """
    yscores = []
    X = zscore(X)
    y = np.array(y)
    print "[NFOLD TRAIN TESTING Easy LR]"
    groups = range(len(y))
    outcomes = dict(zip(groups, y))
    groupFolds = stratifyGroups(groups, outcomes, folds, randomState = 42)
    ymap = []
    for testgs in groupFolds:
        testgs = np.array(testgs)
        setgs = set(testgs)
        traings = [i for i in groups if i not in setgs]
        ytest = y[testgs]
        Xtest = X[testgs]
        ytrain = y[traings]
        Xtrain = X[traings]

        weights = np.array([0.0]*Xtest.shape[1])
        sumWeights = 0.0
        for c in range(Xtrain.shape[1]):
            weights[c] = ((np.absolute(pos_neg_auc(ytrain, Xtrain[:,c])) - 0.5) / 0.5)**2
            sumWeights += weights[c]
        weights = np.array([weights / sumWeights])
        print weights
        ypred_probs = np.dot(Xtest, weights.T)
        print "   EASY LR AUC: %.4f" %  pos_neg_auc(ytest, ypred_probs[:,-1])
        yscores.extend(ypred_probs)
        ymap.extend(testgs)
    newyscores = [0]*len(y)
    for i in groups:
        newyscores[ymap[i]] = yscores[i]
    print "[Done]"
    print " FINAL EASY LR AUC: %.4f" %  pos_neg_auc(y, np.array(newyscores)[:,-1])
    return newyscores


def paired_t_1tail_on_errors(newprobs, oldprobs, ytrue):
    #checks if 
    n = len(ytrue)
    newprobs_res_abs = np.absolute(array(ytrue) - array(newprobs))
    oldprobs_res_abs = np.absolute(array(ytrue) - array(oldprobs))
    ypp_diff = newprobs_res_abs - oldprobs_res_abs #old should be smaller
    ypp_diff_mean, ypp_sd = np.mean(ypp_diff), np.std(ypp_diff)
    ypp_diff_t = ypp_diff_mean / (ypp_sd / sqrt(n))
    ypp_diff_p = ss.t.sf(np.absolute(ypp_diff_t), n-1)
    return ypp_diff_t, ypp_diff_p