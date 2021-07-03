import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import StratifiedShuffleSplit
from ..utils import standardize_headers

HEADERS = ['MPG', 'Cylinder', 'Displacement', 'Horsepower', 'Weight',
           'Acceleration', 'Model Year', 'Origin']

PROJECT_DIR = Path(__file__).parents[2]
TEST_SIZE = 0.2
TARGET_VARIABLE = 'mpg'
RANDOM_STATE = 42

def load_data():
    data_file = os.path.join(PROJECT_DIR, 'data', 'auto-mpg.data')
    df = pd.read_csv(data_file, names=HEADERS, na_values='?',
                 comment = '\t', sep=' ', skipinitialspace=True)

    df.columns = standardize_headers(df.columns)

    return df



splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def split_into_train_and_test(df, stratify='cylinder'):
    for idx_train, idx_test in splitter.split(df, df[stratify]): # splitting so that both groups have same distribution of cylinders
        dfX_train, dfy_train = df.loc[idx_train].drop([TARGET_VARIABLE], axis=1), df.loc[idx_train, [TARGET_VARIABLE]]
        dfX_test, dfy_test = df.loc[idx_test].drop([TARGET_VARIABLE], axis=1), df.loc[idx_test, [TARGET_VARIABLE]]

    return dfX_train, dfy_train, dfX_test, dfy_test






