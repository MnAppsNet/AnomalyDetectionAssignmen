import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
from transformations import createSequences

class Models:

    class MultiModelTypes:
        #This determines how to use the trained models for score calculation
        #when we have multiple models
        CrossModels = "CROSS"     #Use all models to make a prediction for a record and get the mean score
        SingleModel = "SINGLE"    #Use a different single model for each series

    ONE_CLASS_SVM = "OneClassSVM"
    ISOLATION_FOREST = "IsolationForest"
    LOCAL_OUTLIER_FACTOR = "LocalOutlierFactor"
    LONG_SHORT_TERM_MEMORY = "LSTM"
    

    def getModel(method):
        model = None
        unsupervised = True

    #
    # >> LocalOutlierFactor - Semi-Supervised
    #
        if method == Models.LOCAL_OUTLIER_FACTOR:
            unsupervised = False
            model = LocalOutlierFactor(n_neighbors = 20, metric="minkowski", p=2, novelty=True)
            #Minkowski distance: Sum_i( |X_i - y_i|^p )^(1/p)

    #
    # >> LocalOutlierFactor - Unsupervised
    #
        elif method == Models.ISOLATION_FOREST:
            model = IsolationForest(n_estimators=100,n_jobs=-1)

    #
    # >> LocalOutlierFactor - Unsupervised
    #
        elif method == Models.ONE_CLASS_SVM:
            model = OneClassSVM(kernel='linear')

    #
    # >> LocalOutlierFactor - Semi-Supervised
    #
        elif method == Models.LONG_SHORT_TERM_MEMORY:
            unsupervised = False
            class LSTM_Anomaly_Detector:
                def __init__(self,timesteps=5):
                    self.timesteps = timesteps
                
                def fit(self,x):
                    x_train, y_train = createSequences(x, self.timesteps)
                    self.model = Sequential()
                    self.model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
                    self.model.add(Dense(x_train.shape[2]))
                    self.model.compile(optimizer='adam', loss='mean_squared_error')
                    history = self.model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1, shuffle=False)
                    return self
                
                def decision_function(self,X):
                    x, y = createSequences(X, self.timesteps)
                    x_pred = self.model.predict(x, verbose=0)
                    loss = np.mean(np.abs(x_pred - X[self.timesteps:]), axis=1)
                    scores = [0 for i in range(self.timesteps)]
                    scores.extend(loss.tolist())
                    return scores
            
            model = LSTM_Anomaly_Detector()

        return model, unsupervised