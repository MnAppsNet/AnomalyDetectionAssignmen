import numpy as np, tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
from tensorflow.keras.callbacks import Callback


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

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
    

    def getModel(method,window):
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
                def __init__(self, window = 10):
                    self._features_num = 19
                    self.window = window
                    self.encoder = Sequential([
                        Dense(128, activation="relu",input_shape=(self.window*self._features_num,)),
                        Dense(256)
                    ])
                    self.decoder = Sequential([
                        Dense(256, activation="relu"),
                        Dense(128, activation="relu"), 
                        Dense(self.window*self._features_num)
                    ])

                    self.autoencoder = Sequential([
                        self.encoder,
                        self.decoder
                    ])

                    self.autoencoder.compile(loss="mse")

                    self.autoencoder.summary()

                def divide_data(self,x):
                    data = []
                    i = 0
                    while i + self.window < len(x):
                        data.append(x[i:i+self.window])
                        i += 1
                    return np.array(data).reshape(-1,self.window*self._features_num)

                def fit(self,x):
                    data = self.divide_data(x)

                    self.autoencoder.fit(
                        x=data,
                        y=data,
                        validation_split=0.2,
                        epochs=100,
                        callbacks=[EarlyStoppingByLossVal(monitor='val_loss', value=0.01, verbose=1)] )
                    return self
                
                def decision_function(self,X):
                    data = self.divide_data(X)
                    reconstracted_x = np.array(self.autoencoder(data))
                    scores = [[] for i in range(len(X))]
                    j = 0
                    for i in range(len(reconstracted_x)):
                        k = j
                        for l in range(0,len(reconstracted_x[i]),19):
                            score = np.sqrt(metrics.mean_squared_error(data[i][l:l+19], reconstracted_x[i][l:l+19]))
                            scores[k].append(np.mean(score))
                            k += 1
                        j += 1
                    for i in range(len(scores)):
                        if len(scores[i]) == 0:
                            del scores[i]
                        else:
                            scores[i] = np.mean(scores[i])
                    
                    return (scores - min(scores)) / (max(scores) - min(scores))
            
            model = Auto_Encoder(window)
    #
    # >> LSTM - Semi-Supervised
    #
        elif method == Models.LONG_SHORT_TERM_MEMORY:
            unsupervised = False
            class LSTM_Anomaly_Detector:
                def __init__(self,window=5):
                    self.window = window

                def createSequences(self,x):
                    X, Y = [], []
                    for i in range(self.window,len(x)):
                        X.append(x[i-self.window:i])
                        Y.append([x[i]])
                    return np.array(X),np.array(Y)

                def fit(self,x):
                    x_train, y_train = self.createSequences(x)
                    # model definition
                    self.model = Sequential([
                        LSTM(128, activation='tanh', input_shape=(x_train.shape[1],x_train.shape[2])),
                        Dense(19)
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
                    x, _ = self.createSequences(X)
                    x_pred = self.model.predict(x, verbose=0)
                    scores = [0 for i in range(self.window)]
                    for i in range(len(x_pred)):
                        loss = (x_pred[i] - X[self.window+i])**2
                        scores.append(sum(loss))

                    mean_score = np.mean(scores)
                    for i in range(self.window):
                        scores[i] = mean_score

                    return (scores - min(scores)) / (max(scores) - min(scores))
            
            model = LSTM_Anomaly_Detector(window)

        return model, unsupervised