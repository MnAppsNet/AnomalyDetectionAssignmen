import numpy as np, tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
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
    AUTO_ENCODER = "AutoEncoder"
    

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
    # >> OnceClassSVM - Unsupervised
    #
        elif method == Models.ONE_CLASS_SVM:
            model = OneClassSVM(kernel='linear')

    #
    # >> AutoEncoder - Semi-Supervised
    #
        elif method == Models.AUTO_ENCODER:
            unsupervised = False
            class Auto_Encoder:
                def __init__(self):
                    self.encoder = Sequential([
                        Dense(19, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(64)
                    ])

                    self.decoder = Sequential([
                        Dense(64, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(19) # decode to two dimensions again
                    ])

                    self.autoencoder = Sequential([
                        self.encoder,
                        self.decoder
                    ])

                    self.autoencoder.compile(loss="mse")
                
                def fit(self,x):
                    self.autoencoder.fit(
                        x=x,
                        y=x,
                        validation_split=0.2,
                        epochs=100
                    )
                    return self
                
                def decision_function(self,X):
                    reconstracted_x = np.array(self.autoencoder(X))
                    scores = []
                    for i in range(len(reconstracted_x)):
                        score = (reconstracted_x[i] - X[i])**2
                        scores.append(sum(score))
                    
                    return (scores - min(scores)) / (max(scores) - min(scores))
            
            model = Auto_Encoder()
    #
    # >> LSTM - Semi-Supervised
    #
        elif method == Models.LONG_SHORT_TERM_MEMORY:
            unsupervised = False
            class LSTM_Anomaly_Detector:
                def __init__(self,timesteps=5):
                    self.timesteps = timesteps
                
                def fit(self,x):
                    x_train, y_train = createSequences(x, self.timesteps)
                    # model definition
                    self.model = Sequential([
                        LSTM(128, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])),
                        Dense(x_train.shape[2])
                        ])
                    self.model.compile(optimizer='adam', loss='mae')
                    print(self.model.summary())
                    history = self.model.fit(
                        x_train, y_train, 
                        epochs=10, 
                        batch_size=32, 
                        validation_split=0.1,
                        shuffle=False)
                    return self
                
                def decision_function(self,X):
                    x, _ = createSequences(X, self.timesteps)
                    x_pred = self.model.predict(x, verbose=0)
                    scores = [0 for i in range(self.timesteps)]
                    for i in range(len(x_pred)):
                        loss = (x_pred[i] - X[self.timesteps+i])**2
                        scores.append(sum(loss))

                    mean_score = np.mean(scores)
                    for i in range(self.timesteps):
                        scores[i] = mean_score

                    return (scores - min(scores)) / (max(scores) - min(scores))
            
            model = LSTM_Anomaly_Detector(10)

        return model, unsupervised