import numpy as np

def createSequences(x, timesteps):
    X, Y = [], []
    for i in range(len(x)-timesteps):
        X.append(x[i:(i+timesteps)])
        Y.append(x[i+timesteps])
    return np.array(X), np.array(Y)