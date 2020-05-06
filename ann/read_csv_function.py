import pandas as pd


def read_csv(filename):
    y = pd.read_csv(filename, header=None, usecols=[0])
    X = pd.read_csv(filename, header=None)
    X = X.drop(X.columns[0], axis=1)
    return X, y
