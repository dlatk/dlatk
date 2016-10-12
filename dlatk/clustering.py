"""
Clustering

Interfaces with dlatk and scikit-learn
to perform prediction of outcomes for language features.
"""

from .fwConstants import warn
import pickle as pickle

try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.rinterface import RNULLType
except ImportError:
    warn("rpy2 cannot be imported")
    pass

import pandas as pd

from inspect import ismethod
import sys
import random
from itertools import combinations

# scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RandomizedLasso
from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit, train_test_split
from sklearn.decomposition import RandomizedPCA, MiniBatchSparsePCA, PCA, KernelPCA, NMF, SparsePCA
from sklearn.grid_search import GridSearchCV 
from sklearn import metrics
from sklearn.feature_selection import f_regression, SelectPercentile, SelectKBest
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model.base import LinearModel

from scipy.stats import zscore
from scipy.stats.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
import numpy as np
from numpy import sqrt, outer
from numpy.linalg import norm
import math

from .fwConstants import alignDictsAsX

def alignDictsAsXy(X, y, sparse=False, returnKeyList=False):
    """turns a list of dicts for x and a dict for y into a matrix X and vector y"""
    keys = frozenset(list(y.keys()))
    if sparse:
        keys = keys.intersection([item for sublist in X for item in sublist])  # union X keys
    else:
        keys = keys.intersection(*[list(x.keys()) for x in X])  # intersect X keys
    keys = list(keys)  # to make sure it stays in order
    listy = [y[k] for k in keys]
    if sparse:
        keyToIndex = dict([(keys[i], i) for i in range(len(keys))])
        row = []
        col = []
        data = []
        for c in range(len(X)):
            column = X[c]
            for keyid, value in column.items():
                if keyid in keyToIndex:
                    row.append(keyToIndex[keyid])
                    col.append(c)
                    data.append(value)
        sparseX = csr_matrix((data, (row, col)))
        if returnKeyList:
            return (sparseX, listy, keys)
        else:
            return (sparseX, listy)

        
    else: 
        listX = [[x[k] for x in X] for k in keys]
        if returnKeyList:
            return (listX, listy, keys)
        else:
            return (listX, listy)



class DimensionReducer:
    """Handles clustering of continuous outcomes"""

    # feature selection:
    featureSelectionString = None
    # featureSelectionString = 'RandomizedLasso(random_state=42, n_jobs=self.cvJobs)'
    # featureSelectionString = 'ExtraTreesRegressor(n_jobs=self.cvJobs, random_state=42, compute_importances=True)'
    # featureSelectionString = 'SelectKBest(f_regression, k=int(len(y)/3))'
    # featureSelectionString = 'SelectPercentile(f_regression, 33)'#1/3 of features
    # featureSelectionString = \
    #    'Pipeline([("univariate_select", SelectPercentile(f_regression, 33)), ("L1_select", RandomizedLasso(random_state=42, n_jobs=self.cvJobs))])

    # dimensionality reduction (TODO: make a separate step than feature selection)
    # featureSelectionString = 'RandomizedPCA(n_components=min(int(X.shape[1]*.10), int(X.shape[0]/2)), random_state=42, whiten=False, iterated_power=3)'
    # featureSelectionString = 'PCA(n_components=min(int(X.shape[1]*.02), X.shape[0]), whiten=False)'
    # featureSelectionString = 'PCA(n_components=None, whiten=False)'
    # featureSelectionString = 'KernelPCA(n_components=int(X.shape[1]*.02), kernel="rbf", degree=3, eigen_solver="auto")'  
    # featureSelectionString = \
    #    'MiniBatchSparsePCA(n_components=int(X.shape[1]*.05), random_state=42, alpha=0.01, chunk_size=20, n_jobs=self.cvJobs)'

    featureSelectPerc = 1.00  # only perform feature selection on a sample of training (set to 1 to perform on all)
    # featureSelectPerc = 0.20 #only perform feature selection on a sample of training (set to 1 to perform on all)

    testPerc = .20  # percentage of sample to use as test set (the rest is training)
    randomState = 42  # percentage of sample to use as test set (the rest is training)
    
    params = {
            #'nmf' : { 'n_components': 15, 'init': 'nndsvd', 'sparseness': None, 'beta': 1, 'eta' : 0.1, 'tol': .0001, 'max_iter' : 200, 'nls_max_iter': 2000, 'random_state' :42 },
            'nmf' : { 'n_components': 15, 'init': 'nndsvd', 'solver':'cd', 'l1_ratio': 0.95, 'alpha': 10, 'max_iter' : 200, 'nls_max_iter': 2000, 'random_state' :42 },

            'pca' : { 'n_components': 'mle', 'whiten': False},
            #'pca' : { 'n_components': 'mle', 'whiten': True},

            #'sparsepca': {'n_components':None, 'alpha':1, 'ridge_alpha':0.01, 'method': 'lars', 'n_jobs':4, 'random_state':42},
            'sparsepca': {'n_components':None, 'alpha':1, 'ridge_alpha':0.01, 'method': 'cd', 'n_jobs':4, 'random_state':42},
            
            'lda': { 'nb_topics':50, 'dictionary':None, 'alpha':None },

            'rpca': {'n_components':15, 'random_state':42, 'whiten':False, 'iterated_power':3},

            }
    # maps the identifier of the algorithm used to the actual class name from the module
    modelToClassName = {
        'nmf' : 'NMF',
        'pca' : 'PCA',
        'sparsepca': 'SparsePCA',
        'lda' : 'LDA',
        'rpca' : 'RandomizedPCA',
        }

    def __init__(self, fg, modelName='nmf', og=None):
        # initialize regression predictor
        self.outcomeGetter = og
        self.featureGetter = fg
        self.modelName = modelName

        # object that stores most of the model data
        self.clusterModels = dict()

        # scales the data in the matrix
        self.scalers = dict()
        
        # selects appropriate features/columns
        self.fSelectors = dict()

        
        self.featureNames = []  # holds the order the features are expected in

    def fit(self, standardize=True, sparse=False, restrictToGroups=None):
        """Create clusters"""
        # restrictToGroups: list of groups to which the algorithm should restrict itself

        print()
        # 1. get data possible ys (outcomes)
        groups = []
        #list of control values for groups
        controlValues = None
        allOutcomes = None
        if self.outcomeGetter != None: # and self.outcomeGetter.hasOutcomes():
            (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
            if restrictToGroups:  # restrict to groups 
                groups = groups.intersection(restrictToGroups)
                for outcomeName, outcomes in allOutcomes.items():
                    allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
                for controlName, controlValues in controls.items():
                    controls[controlName] = dict([(g, controlValues[g]) for g in groups])
            print("[number of groups: %d]" % len(groups))
            controlValues = list(controls.values())  # list of dictionaries of group=>group_norm
        elif restrictToGroups:
            print("[Not using outcomes]")
            groups = restrictToGroups

        # 2. get data for X:
        (groupNorms, featureNames) = (None, None)
        if sparse:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsSparseFeatsFirst(groups)
        else:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)

        self.featureNames = list(groupNorms.keys())  # holds the order to expect features
        groupNormValues = list(groupNorms.values())  # list of dictionaries of group => group_norm
        
    #     this will return a dictionary of dictionaries

        # 3. Create models for each possible y:
        if allOutcomes:
            for outcomeName, outcomes in allOutcomes.items():
                print("\n= %s =\n%s" % (outcomeName, '-' * (len(outcomeName) + 4)))
                print("[Aligning Dicts to get X and y]")
                (X, y) = alignDictsAsXy(groupNormValues + controlValues, outcomes, sparse)
                (self.clusterModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName]) = self._fit(X, y, standardize)
        else:
            X = alignDictsAsX(groupNormValues + controlValues, sparse, returnKeyList= False)
            (self.clusterModels['noOutcome'], self.scalers['noOutcome'], self.fSelectors['noOutcome']) = self._fit(X, standardize)

    def _fit(self, X, y=[], standardize=True):
        """does the actual regression training, can be used by both train and test"""

        sparse = True
        if not isinstance(X, csr_matrix):
            X = np.array(X)
            sparse = False
     
        scaler = None
        if standardize == True:
            scaler = StandardScaler(with_mean=not sparse)
            print("[Applying StandardScaler to X: %s]" % str(scaler))
            X = scaler.fit_transform(X)

        if 'nmf' in self.modelName.lower():
            minX = X.min()
            X = X + (minX*-1)

        #if y:
        #    y = np.array(y)
        
        print(" (N, features): %s" % str(X.shape))

        fSelector = None
        if self.featureSelectionString:
            fSelector = eval(self.featureSelectionString)
            print("[Applying Feature Selection to X: %s]" % str(fSelector))
            X = fSelector.fit_transform(X, y)
            print(" after feature selection: (N, features): %s" % str(X.shape))

           
#        if hasMultValuesPerItem(self.cvParams[self.modelName.lower()]) and self.modelName.lower()[-2:] != 'cv':
#            #grid search for classifier params:
#            gs = GridSearchCV(eval(self.modelToClassName[self.modelName.lower()]+'()'), 
#                              self.cvParams[self.modelName.lower()], n_jobs = self.cvJobs)
#            print "[Performing grid search for parameters over training]"
#            gs.fit(X, y, cv=ShuffleSplit(len(y), n_iterations=(self.cvFolds+1), test_size=1/float(self.cvFolds), random_state=0))
#
#            print "best estimator: %s (score: %.4f)\n" % (gs.best_estimator_, gs.best_score_)
#            return gs.best_estimator_, scaler, fSelector
#        else:
        # no grid search
        print("[Doing clustering using : %s]" % self.modelName.lower())
        cluster = eval(self.modelToClassName[self.modelName.lower()] + '()')
        if 'lda' in self.modelName.lower():
            self.params['lda']['dictionary'] = self.featureNames        
        cluster.set_params( **self.params[self.modelName.lower()])

        if y:
            cluster.fit(X,y)
        else:
            cluster.fit(X)
            
#        print "coefs"
#        print cluster.coef_
        print("model: %s " % str(cluster))

        return cluster, scaler, fSelector
    
    def transform(self, standardize=True, sparse=False, restrictToGroups=None):
        groups = []
        controlValues = None
        allOutcomes = None
        if self.outcomeGetter != None:
            (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
            if restrictToGroups:  # restrict to groups
                groups = groups.intersection(restrictToGroups)
                for outcomeName, outcomes in allOutcomes.items():
                    allOutcomes[outcomeName] = dict([(g, outcomes[g]) for g in groups if (g in outcomes)])
                for controlName, controlValues in controls.items():
                    controls[controlName] = dict([(g, controlValues[g]) for g in groups])
            print("[number of groups: %d]" % len(groups))
            controlValues = list(controls.values())  # list of dictionaries of group=>group_norm
        elif restrictToGroups:
            groups = restrictToGroups

        # 2. get data for X:
        (groupNorms, featureNames) = (None, None)
        if sparse:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsSparseFeatsFirst(groups)
        else:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)

        groupNormValues = []
        transformGroups = None
        print("[Aligning current X with training X]\n")
        for feat in self.featureNames:
            if feat in groupNorms:
                groupNormValues.append(groupNorms[feat])
            else:
                if sparse: #assumed all zeros
                    groupNormValues.append(dict())
                else: #need to insert 0s
                    if not transformGroups:
                        transformGroups = list(groupNorms[next(iter(groupNorms.keys()))].keys())
                    groupNormValues.append(dict([(k, 0.0) for k in transformGroups]))
        #groupNormValues = groupNorms.values() #list of dictionaries of group => group_norm
        print("number of features after alignment: %d" % len(groupNormValues))
        transformedX = dict()
        group_ids = []
        if allOutcomes:
            for outcomeName, outcomes in allOutcomes.items():
                print("\n= %s =\n%s"%(outcomeName, '-'*(len(outcomeName)+4)))
                X, group_ids = alignDictAsX(groupNormValues + controlValues, sparse, returnKeyList=True)
                (cluster, scaler, fSelector) = (self.clusterModels[outcomeName], self.scalers[outcomeName], self.fSelectors[outcomeName])
                transformedX[outcomeName] = _transform(cluster, scaler, fSelector)
        else:
            X, group_ids = alignDictAsX(groupNormValues + controlValues, sparse, returnKeyList=True)
            (cluster, scaler, fSelector) = (self.clusterModels['noOutcome'], self.scalers['noOutcome'], self.fSelectors['noOutcome'])
            transformedX['noOutcome'] = _transform(cluster, scaler, fSelector)
            
        for outcomeName, outcomeX  in transformedX.items():
            if not isinstance(outcomeX, csr_matrix):
                dictX = dict()
                (n, m) = outcomeX.shape
                for j in range(m):
                    dictX[group_ids[j]] = dict()
                    for i in range(n):
                        dictX[group_ids[j]]['rfeat'+str(i)]= outcomeX[i][j]
                transformedX[outcomeName] = dictX
            else:
                raise NotImplementedError
        
        return transformedX
    
    def _transform(self, cluster, X, scaler = None, fSelector = None, y = None):
        if scaler:
            X = scaler.transform(X)
        if fSelector:
            X = fSelector.transform(X)
        
        return cluster.transform(X)
    
    def modelToLexicon(self):
        lexicons = dict()
        for outcomeName, model in self.clusterModels.items():
            reduction_dict = dict()
            if self.fSelectors[outcomeName]:
                print('Error: Does not handle writing models with feature selection to a lexicon (yet)')
                raise NotImplementedError
            component_mat = model.components_
            (n,m) = component_mat.shape
            print('components shape : %s', str(component_mat.shape))
            for i in range(n):
                reduction_dict['rfeat'+str(i)] = dict()
                for j in range(m):
                     if component_mat[i][j] > 0:
                         #print "feature name: %s"% self.featureNames[j] 
                         reduction_dict['rfeat'+str(i)][self.featureNames[j]] = component_mat[i][j]
            lexicons[outcomeName] = reduction_dict
        return lexicons
        
    ######################
    def load(self, filename):
        print("[Loading %s]" % filename)
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    def save(self, filename):
        print("[Saving %s]" % filename)
        f = open(filename, 'wb')
        toDump = {'modelName': self.modelName,
                  'clusterModels': self.clusterModels,
                  'scalers' : self.scalers,
                  'fSelectors' : self.fSelectors,
                  'featureNames' : self.featureNames
                  }
        pickle.dump(toDump, f, 2)
        f.close()

#class LDA (BaseReducerClass)

def chunks(X, y, size):
    """ Yield successive n-sized chunks from l."""
    if not isinstance(X, csr_matrix):
        assert len(X) == len(y), "chunks: size of X and y don't match"
    size = max(len(y), size)

    
    for i in range(0, len(y), size):
        yield X[i:i + size], y[i:i + size]

def hasMultValuesPerItem(listOfD):
    """returns true if the dictionary has a list with more than one element"""
    if len(listOfD) > 1:
        return True
    for d in listOfD:
        for value in d.values():
            if len(value) > 1: return True
    return False

def r2simple(ytrue, ypred):
    y_mean = sum(ytrue) / float(len(ytrue))
    ss_tot = sum((yi - y_mean) ** 2 for yi in ytrue)
    ss_err = sum((yi - fi) ** 2 for yi, fi in zip(ytrue, ypred))
    r2 = 1 - (ss_err / ss_tot)
    return r2



class CCA:
    """Handles CCA analyses of language and outcomes"""
    
    def __init__(self, fg, og, numComponents=15):
        try:
            import pandas.rpy.common as com
        except ImportError:
            warn("pandas.rpy.common cannot be imported")
            pass
        # initialize CCA object
        self.outcomeGetter = og
        self.featureGetter = fg
        self.numComponents = numComponents
        self.model = {'u': None, # Pandas DF so we have the list of outcomes and topics
                      'v': None, # Pandas DF so we have the list of outcomes and topics
                      'd': None  # Vector
                  }

    def saveModel(self,filename):
        print("Saving model to %s" % filename)
        with open(filename, "wb+") as f:
            pickle.dump(self.model, f)

    def loadModel(self, filename):
        print("Loading model from %s" % filename)
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        
    def RsoftImpute(self, X):
        softImpute = importr("softImpute")
        X = com.convert_to_r_dataframe(X)
        X = softImpute.complete(X,softImpute.softImpute(softImpute.biScale(X, maxit = 100)))
        X = com.convert_robj(X)
        return X

    def prepMatricesTogether(self, X, Z, NAthresh = 4):
        """\tConcatenates X and Z, then completes the resulting matrix, splits the matrix back"""

        print("Performing SoftImpute on the concatenated controls+outcomes matrix")
        # Concatenate matrices together
        Zcols = Z.columns
        Xcols = X.columns

        Z = pd.concat([Z,X], axis=1)

        ## Cleaning the outcome table
        ## removing counties that have less than NAthresh4 diseases
        Z = Z[Z.apply(lambda x: sum(1 for i in x if not np.isnan(i)), axis=1) >= NAthresh]

        Zfreqs = Z[Zcols].apply(lambda x: sum(1 for i in x if not np.isnan(i)))
        
        Z = self.RsoftImpute(Z)
        Z.columns = [z.strip("X") for z in Z.columns]

        Xfreqs = X.apply(lambda x: sum(1 for i in x if not np.isnan(i)))

        Xfreqs.index = X.columns

        X = Z[Xcols]
        Z = Z[Zcols]
 
        Zfreqs.index = Z.columns
        
        Ngroups = X.shape[0]

        return X, Z, Xfreqs.to_dict(), Zfreqs.to_dict()
        
    def prepMatrices(self, X, Z, NAthresh = 4, softImputeXtoo = False, softImputeXZtogether = False):
        """Completes matrices that are incomplete, imputes rows that don't have enough data (NAthresh), and aligns the rows"""

        #with open("countyDiseaseData.pickle", "wb+") as f:
        #    pickle.dump((X,Z), f)
        #    print "Dumped data to countyDiseaseData.pickle"

        if softImputeXZtogether:
            # Concatenate matrices together
            oldZ = Z
            oldX = X
            Z = pd.concat([Z,X], axis=1)

        ## Cleaning the outcome table
        ## removing counties that have less than NAthresh4 diseases
        Z = Z[Z.apply(lambda x: sum(1 for i in x if not np.isnan(i)), axis=1) >= NAthresh]
        Zfreqs = Z.apply(lambda x: sum(1 for i in x if not np.isnan(i)))
        
        Z = self.RsoftImpute(Z)


        # Removing groups that didn't make the NAN criterion
        if softImputeXtoo:
            X = X[X.apply(lambda x: sum(1 for i in x if not np.isnan(i)), axis=1) >= NAthresh]
        Xfreqs = X.apply(lambda x: sum(1 for i in x if not np.isnan(i)))
        if softImputeXtoo:
            X = self.RsoftImpute(X)
        Xfreqs.index = X.columns

        X = X[X.index.isin(Z.index)]
        Z = Z[Z.index.isin(X.index)]
 
        Zfreqs.index = Z.columns
        
        Ngroups = X.shape[0]

        return X, Z, Xfreqs.to_dict(), Zfreqs.to_dict()


    def predictCompsToSQL(self,tablename=None,  csv = False, outputname = None, NAthresh = 4, useXmatrix = False):
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        # groups: set(group_ids)
        # allOutcomes: {outcome: {group_id: value}}
        # controls: {control: {group_id: value}}
        dataDict = {}
        if useXmatrix:
            (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)
            # groupNorms: {feat: {group_id: group_norm}}
            dataDict = groupNorms
        else:
            dataDict = allOutcomes
            dataDict.update(controls)

        # Hack, don't use:
        # dataDict = controls

        df = pd.DataFrame(data=dataDict)
        df = df[df.apply(lambda x: sum(1 for i in x if not np.isnan(i)), axis=1) >= 4]
        df = self.RsoftImpute(df)
        comps = self.model['u'] if useXmatrix else self.model['v']

        assert comps.shape[0] == df.shape[1], "Number of outcomes (& controls) is wrong. Please see wiki.wwbp.org on how to fix this."
        C = pd.DataFrame(np.dot(df, comps), index = df.index, columns = comps.columns)

        if csv:
            fn = outputname
            if fn[-4:] != '.csv': fn += '.csv'
            C.to_csv(open(fn,"w+"), index_label = self.outcomeGetter.correl_field)
        if tablename:
            self.outcomeGetter.createOutcomeTable(tablename, C)

        return C
    
    def _cca(self, X, Z, **params):
        """Given two Pandas dataframes and a set of parameters, performs CCA
        returns CCA dict (converted from R CCA named list object)
        """
        
        pma = importr("PMA")
        
        # Defaults:
        kwParams = {"typex": "standard",
                    "typez": "standard",
                    "trace": False,
                    "K": self.numComponents,
        }
        kwParams.update(params)
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = com.convert_to_r_dataframe(X)
        if isinstance(Z, pd.core.frame.DataFrame):
            Z = com.convert_to_r_dataframe(Z)

        assert isinstance(X, ro.vectors.DataFrame) and isinstance(Z, ro.vectors.DataFrame), "X, Z need to be either Pandas DataFrames or R dataframes!"

        assert self.numComponents <= min(len(X.names),len(Z.names)), "Number of components must be smaller than the minimum of columns in each of your matrices"

        nGroups = com.convert_robj(ro.r["nrow"](X)[0])
        
        print("\tCCA parameters:", kwParams)
        cca = pma.CCA(X, Z, **kwParams)
        cca = {k:v for k, v in list(cca.items())}
        cca['nGroups'] = nGroups
        return cca
        
    def ccaOutcomesVsControls(self, penaltyX = None, penaltyZ = None, NAthresh = 4):
        """Performs CCA using controls and outcomes, no language"""
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        # groups: set(group_ids)
        # allOutcomes: {outcome: {group_id: value}}
        # controls: {control: {group_id: value}}

        print("X: controls\nZ: outcomes")
        Zdict = allOutcomes
        Xdict = controls

        # R doesn't handle '$'s in column names
        Xdict = {k.replace('$','.'):v for k, v in Xdict.items()}
        Zdict = {k.replace('$','.'):v for k, v in Zdict.items()}

        Xdf = pd.DataFrame(data=Xdict)
        Zdf = pd.DataFrame(data=Zdict)
        
        # X, Z, Xfreqs, Zfreqs = self.prepMatrices(Xdf,Zdf, NAthresh = NAthresh, softImputeXtoo=True)
        X, Z, Xfreqs, Zfreqs = self.prepMatricesTogether(Xdf,Zdf, NAthresh = NAthresh)
        kwParams = {}
        if penaltyX: kwParams['penaltyx'] = penaltyX
        if penaltyZ: kwParams['penaltyz'] = penaltyZ
        # kwParams['upos'] = True
        # kwParams['vneg'] = True

        cca = self._cca(X,Z, **kwParams)
    
        Xcomp = com.convert_robj(cca['u']) # Controls
        Zcomp = com.convert_robj(cca['v']) # Outcomes

        d = com.convert_robj(cca['d']) # Something
        self.model = {
            'u': Xcomp,
            'v': Zcomp,
            'd': d,
        }

        featureNames = X.columns
        Xcomp.index = [i.strip("X") for i in featureNames]
        Xfreqs = {k.strip("X"): v for k,v in Xfreqs.items()}
        Xcomp.columns = ["%.2d_comp" % i for i in range(Xcomp.shape[1])]

        outcomeNames = Z.columns
        Zcomp.index = [i.strip("X") for i in outcomeNames]
        Zfreqs = {k.strip("X"): v for k,v in Zfreqs.items()}
        Zcomp.columns = ["%.2d_comp" % i for i in range(Zcomp.shape[1])]
        
        Zcomp2 = pd.concat([Xcomp, Zcomp])
        
        Xcomp_dict = {k: {i:(j,
                             0.0 if j != 0 else 1,
                             cca["nGroups"],
                             Xfreqs[i]) for i, j in v.items()} for k, v in Xcomp.to_dict().items()}
        Zcomp_dict = {k: {i:(j,0.0 if j != 0 else 1,cca["nGroups"],
                             Zfreqs[i] if i in list(Zfreqs.keys()) else Xfreqs[i]
                         ) for i, j in v.items()} for k, v in Zcomp2.to_dict().items()}

        d_dict = dict(list(zip(Zcomp.columns,d)))
        return Xcomp_dict, Zcomp_dict, d_dict

    def cca(self, penaltyX = None, penaltyZ = None, NAthresh = 4, controlsWithFeats = False):
        """Performs CCA based on the outcomes and controls (X: features, Z: outcomes)"""
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        # groups: set(group_ids)
        # allOutcomes: {outcome: {group_id: value}}
        # controls: {control: {group_id: value}}

        (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)
        # groupNorms: {feat: {group_id: group_norm}}

        Zdict = allOutcomes
        Xdict = groupNorms
        
        if controlsWithFeats:
            print("Appending controls to X")
            Xdict.update(controls)
        else:
            print("Appending controls to Z")
            Zdict.update(controls)        

        # R doesn't handle '$'s in column names
        Xdict = {k.replace('$','.'):v for k, v in Xdict.items()}
        Zdict = {k.replace('$','.'):v for k, v in Zdict.items()}

        # TO DO: get topic frequencies?

        # featureNames: list of possible feature names

        # X contains feature group_norms, Z contains outcome values
        Xdf = pd.DataFrame(data=Xdict)
        Zdf = pd.DataFrame(data=Zdict)

        assert self.numComponents <= min(Xdf.shape[1],Zdf.shape[1]), "Number of components cannot be more than min(#feats, #outcomes+#controls)"
        X, Z, Xfreqs, Zfreqs = self.prepMatrices(Xdf,Zdf, NAthresh = NAthresh)
        
        Xt_Z = pd.DataFrame(np.dot(X.transpose(),Z), index = X.columns, columns = Z.columns)
        
        kwParams = {}
        if penaltyX: kwParams['penaltyx'] = penaltyX
        if penaltyZ: kwParams['penaltyz'] = penaltyZ

        cca = self._cca(X,Z, **kwParams)

        Xcomp = com.convert_robj(cca['u']) # Features
        Zcomp = com.convert_robj(cca['v']) # Outcomes
        d = com.convert_robj(cca['d']) # Something
        self.model = {
            'u': Xcomp,
            'v': Zcomp,
            'd': d,
        }

        with open("/localdata/county-disease/CCA/Xt_Z.Xcomp.Zcomp.d.pickle","wb+") as f:
            print("Dumping data to /localdata/county-disease/CCA/Xt_Z.Xcomp.Zcomp.d.pickle")
            pickle.dump((Xt_Z, Xcomp, Zcomp, d), f)
            print("Dumping data to /localdata/county-disease/CCA/X.Z.pickle")
            pickle.dump((X,Z), f)
        
        reconstruction_err = [ 
            sum(np.linalg.norm(np.outer(Xcomp[i]*d_i,Zcomp[i].transpose()),axis=0))/sum(np.linalg.norm(Xt_Z, axis=0))
            for i, d_i in enumerate(d)
        ]
        print(reconstruction_err)
        
        featureNames = X.columns
        Xcomp.index = [i.strip("X") for i in featureNames]
        Xfreqs = {k.strip("X"): v for k,v in Xfreqs.items()}
        Xcomp.columns = ["%.2d_comp" % i for i in range(Xcomp.shape[1])]

        outcomeNames = Z.columns
        Zcomp.index = [i.strip("X") for i in outcomeNames]
        Zfreqs = {k.strip("X"): v for k,v in Zfreqs.items()}
        Zcomp.columns = ["%.2d_comp" % i for i in range(Zcomp.shape[1])]
        
        d_dict = dict(list(zip(Zcomp.columns,d)))
                
        Xcomp_dict = {k: {i:(j,
                             0.0 if j != 0 else 1,
                             cca["nGroups"],
                             Xfreqs[i]) for i, j in v.items()} for k, v in Xcomp.to_dict().items()}
        Zcomp_dict = {k: {i:(j,0.0 if j != 0 else 1,cca["nGroups"],Zfreqs[i]) for i, j in v.items()} for k, v in Zcomp.to_dict().items()}

        return Xcomp_dict, Zcomp_dict, d_dict
        ## output: {outcome: feat: (r,p,n,freq)}

    def _ccaPermute(self, X, Z, **params):
        """Performs CCA.permute from the PMA package to see which penalty values are better"""
        pma = importr("PMA")
        
        kwParams = {"typex": "standard",
                    "typez": "standard",
                    "trace": True}
        kwParams.update(params)

        print("\tCCA permute parameters:", kwParams)
        
        cca_permute = ro.r['CCA.permute'](X, Z, **kwParams)
        header = ['penaltyxs', 'penaltyzs', 'zstats', 'pvals','cors', 'ft.corperms', 'nnonzerous', 'nnonzerovs']
        header2 = ["X Penalty", "Z Penalty", "Z-Stat", "P-Value", "Cors", "FT(Cors)", "# U's Non-Zero", "# Vs Non-Zero"]

        cca_permute = {k:v for k,v in list(cca_permute.items())}

        df = pd.DataFrame({h:com.convert_robj(cca_permute[h]) for h in header}, columns=header)
        df.columns = header2
        df.index = range(1,18)

        print("\n", df)
        print() 
        print("Best L1 bound for x: %.5f" % com.convert_robj(cca_permute["bestpenaltyx"])[0])
        print("Best L1 bound for z: %.5f" % com.convert_robj(cca_permute["bestpenaltyz"])[0])

    def ccaPermuteOutcomesVsControls(self, nPerms = 25, penaltyXs = None , penaltyZs = None):
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        # groups: set(group_ids)
        # allOutcomes: {outcome: {group_id: value}}
        # controls: {control: {group_id: value}}

        # X contains feature group_norms, Z contains outcome values
        Zdict = allOutcomes
        Xdict = controls
        
        # R doesn't handle '$'s in column names
        Xdict = {k.replace('$','.'):v for k, v in Xdict.items()}
        Zdict = {k.replace('$','.'):v for k, v in Zdict.items()}
        
        # X, Z, Xfreqs, Zfreqs = self.prepMatrices(pd.DataFrame(data=Xdict),pd.DataFrame(data=Zdict), softImputeXtoo=True)
        X, Z, Xfreqs, Zfreqs = self.prepMatricesTogether(pd.DataFrame(data=Xdict), pd.DataFrame(data=Zdict))
        X = com.convert_to_r_dataframe(X)
        Z = com.convert_to_r_dataframe(Z)

        Ngroups = com.convert_robj(ro.r["nrow"](X)[0])
        
        kwParams = {"nperms": nPerms}
        kwParams['penaltyxs'] = penaltyXs if penaltyXs else ro.vectors.FloatVector(np.arange(.1,.91,.05))
        kwParams['penaltyzs'] = penaltyZs if penaltyZs else ro.vectors.FloatVector(np.arange(.1,.91,.05))
        
        self._ccaPermute(X,Z, **kwParams)
        
    def ccaPermute(self, nPerms = 25, penaltyXs = None , penaltyZs = None, controlsWithFeats = False):
        (groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes()
        # groups: set(group_ids)
        # allOutcomes: {outcome: {group_id: value}}
        # controls: {control: {group_id: value}}
        
        (groupNorms, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups)

        Zdict = allOutcomes
        Xdict = groupNorms
        
        if controlsWithFeats:
            print("Appending controls to X")
            Xdict.update(controls)
        else:
            print("Appending controls to Z")
            Zdict.update(controls)        

        # TO DO: get topic frequencies?

        # groupNorms: {feat: {group_id: group_norm}}
        # featureNames: list of possible feature names

        # X contains feature group_norms, Z contains outcome values
        X, Z, Xfreqs, Zfreqs = self.prepMatrices(pd.DataFrame(data=Xdict),pd.DataFrame(data=Zdict))
        X = com.convert_to_r_dataframe(X)
        Z = com.convert_to_r_dataframe(Z)


        Ngroups = com.convert_robj(ro.r["nrow"](X)[0])
        
        kwParams = {"nperms": nPerms}
        kwParams['penaltyxs'] = penaltyXs if penaltyXs else ro.vectors.FloatVector(np.arange(.1,.91,.05))
        kwParams['penaltyzs'] = penaltyZs if penaltyZs else ro.vectors.FloatVector(np.arange(.1,.91,.05))
        
        self._ccaPermute(X,Z, **kwParams)
