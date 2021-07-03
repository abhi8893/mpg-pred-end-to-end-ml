import re
import pandas as pd

def remove_extra_whitespace(s):
    '''remove extra whitespace from string'''
    
    return re.sub(r'\s+', r' ', s.strip())

# TODO: Have option to lemmatize
def standardize_headers(headers):
    '''
    standardize the headers by removing extra whitespace
    and converting to snake_case
    '''
    headers = map(remove_extra_whitespace, headers)
    headers = map(lambda s: re.sub(r' ', '_', s).lower(), headers)
    return list(headers)


def assure_index(df):
    '''assure index is a RangeIndex with step 1'''
    
    df.index = pd.RangeIndex(start=0, stop=df.shape[0])
    return df