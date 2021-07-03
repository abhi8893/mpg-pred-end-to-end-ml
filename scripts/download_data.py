import wget

URL = r'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
wget.download(URL, r'data/')