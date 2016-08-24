# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# License: 3-clause BSD

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array
#from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from scipy.stats import rankdata

class OccurrenceThreshold(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.
    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
 
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
        
    Attributes
    ----------
    `counts_` : array, shape (n_features,)
    Number of  non-zero observations of individual features.
    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold:
                                                                                                                                                                                                              
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]                                                                                                                                                    
        >>> selector = VarianceThreshold()                                                                                                                                                                    
        >>> selector.fit_transform(X)                                                                                                                                                                         
        array([[2, 0],                                                                                                                                                                                        
               [1, 4],                                                                                                                                                                                        
               [1, 1]])                                                                                                                                                                                       
    """

    def __init__(self, threshold=1.0):
        """
        zero_value identify the value that represents 0 when standardizing 
        """
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn the occurrences. good for frequency / count data                                                                                                                                                                                                              
        Parameters
        ----------                                                                                                                                                                                            
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.                                                                                                                                                                                                              
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.                                                                                                                                                                                                              
        Returns
        -------
        self                                                                                                                                                                                                  
        """
        X = check_array(X, dtype=np.float64)

        # if hasattr(X, "toarray"):   # sparse matrix
        #     if self.zero_value != "min" and self.zero_value != 0:
        #         raise ValueError("zero_value other than -0 or min not possible with sparse matrices")
        #     self.avgValues_ = np.diff(X.tocsr().indptr)
        self.means_ = np.mean(X, 0)

        self.ranks_ = self.means_.shape[0] - rankdata(self.means_, method='ordinal')
        if isinstance(self.threshold, float):
            self.threshold = int(round(X.shape[0]*self.threshold))
        print("SET THRESHOLD %s" % self.threshold) #debug

        #print "RANKS: %s" % str(self.ranks_)[:30]

        return self

    def _get_support_mask(self):
        return self.ranks_ < self.threshold


#OLD approach (like p_occ)
        # if hasattr(X, "toarray"):   # sparse matrix
        #     if self.zero_value != "min" and self.zero_value != 0:
        #         raise ValueError("zero_value other than -0 or min not possible with sparse matrices")
        #     self.counts_ = np.diff(X.tocsr().indptr)
        # else:
        #     if self.zero_value == "min":
        #         X = X - X.min(axis=0)
        #     else:
        #         X = X - self.zero_value
        #     self.counts_ = np.apply_along_axis(np.count_nonzero, axis = 0, arr = X)


        # if self.threshold < 1:
        #     self.threshold = long(round(counts_.shape[0]*self.threshold))

        # if np.all(self.counts_ < self.threshold):
        #     msg = "No feature in X meets the occurrence threshold {0:d}"
        #     if X.shape[0] == 1:
        #         msg += " (X contains only one sample)"
        #     raise ValueError(msg.format(self.threshold))
        
