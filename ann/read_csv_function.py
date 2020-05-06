import pandas as pd


def read_csv(filename):
    y = pd.read_csv(filename, header=None, usecols=[0])
    temp = []
    for element in y.values:
        for celement in element:
            temp.append(celement)

    X = pd.read_csv(filename, header=None)
    print(len(temp))
    X = X.drop(X.columns[0], axis=1)
    return X, temp
