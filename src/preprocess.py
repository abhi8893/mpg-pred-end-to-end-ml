from sklearn.base import BaseEstimator, TransformerMixin 
import pandas as pd

class ValueMapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, mapper: dict):
        self.mapper = mapper

    def fit(self, X, y=None):
        # TODO: Get column wise default mappings
        return self

    def transform(self, X):
        X = X.apply(lambda col: col.map(self.mapper)).to_numpy()
        return X