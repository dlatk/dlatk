from sklearn.decomposition import PCA 

class WrappedPCA(PCA):
    """Wrapper around sklearn.decomposition.PCA to maintain the right n_components
    when magic_sauce (and other methods from the same family) is used for feature selection.

    All methods and attributes are inherited from sklearn.decomposition.PCA, 
    except _fit() which is overridden here. For details refer 
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_pca.py
    """

    def _fit(self, X): 
    
        n_samples, n_features = X.shape
        self.n_components = min(self.n_components, min(n_samples, n_features))

        return super()._fit(X)
