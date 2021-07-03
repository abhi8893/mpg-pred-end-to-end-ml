from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd

from ..preprocess import ValueMapper

RANDOM_STATE = 42
COLUMNS = ['cylinder', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year']

origin_mapper = ValueMapper({1: "India", 2: "USA", 3: "Germany"})

class CustomAttrAdder(BaseEstimator, TransformerMixin):

    INDICES = {
        'cylinder': 1,
        'displacement': 2,
        'horsepower': 3,
        'weight': 4,
        'acceleration': 5
    }

    def __init__(self, 
                 on_pairs=(('displacement', 'horsepower'),
                           ('weight', 'cylinder'),
                           ('acceleration', 'horsepower'),
                           ('acceleration', 'cylinder'))):
    
        self.on_pairs = on_pairs
        self.feature_names = []
        for col1, col2 in self.on_pairs:
            self.feature_names.append(f'{col1}_on_{col2}')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = []

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        for col1, col2 in self.on_pairs:
            t = X[:, self.INDICES[col1]]/X[:, self.INDICES[col2]]
            
            transformed.append(t)

        return np.concatenate([X, np.array(transformed).T], axis=1)


cat_encoder = ColumnTransformer([
    ('enc', OneHotEncoder(), ['origin'])
], remainder='drop')

col_dropper = ColumnTransformer([
    ('drop_cols', 'drop', ['origin'])
], remainder='passthrough')

imputer = Pipeline([
    ('drop_cols', col_dropper),
    ('impute', IterativeImputer(random_state=RANDOM_STATE))]
    )

feature_union = FeatureUnion([
    ('imputer', imputer),
    ('cat_encoder', cat_encoder)
    ])

preprocessor = Pipeline([
    ('feature_union', feature_union),
    ('attr_adder', CustomAttrAdder()),
    ('scale', StandardScaler())
])


PREPROCESSED_COLUMNS = COLUMNS  + preprocessor.named_steps['attr_adder'].feature_names + ['origin_1', 'origin_2', 'origin_3']

