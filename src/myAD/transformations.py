import numpy as np

def createSequences(x, timesteps):
    X, Y = [], []
    for i in range(timesteps,len(x)):
        X.append(x[i-timesteps:i])
        Y.append([x[i]])
    return np.array(X),np.array(Y)