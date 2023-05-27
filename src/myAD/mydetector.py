import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from random import choice, randint, random
import tensorflow as tf

class Models:
    ONE_CLASS_SVM = "OneClassSVM"
    ISOLATION_FOREST = "IsolationForest"
    RANDOM_FOREST = "RandomForest"
    RNN = "RecurrentNuralNetwork"

    def getModel(method):
        semiSupervised = False              #Is the method semi supervised?
        history = 0                         #History of records to keep on series reconstruction methods
        model = None

        if method == Models.ONE_CLASS_SVM:
            model = OneClassSVM(gamma='auto')

        elif method == Models.ISOLATION_FOREST:
            model =  IsolationForest(n_estimators=100, max_samples='auto', contamination='auto')

        elif method == Models.RANDOM_FOREST:
            semiSupervised = True
            history = 9
            model = RandomForestRegressor(n_estimators=100,criterion="squared_error")
    
        elif method == Models.RNN:
            semiSupervised = True
            history = 9
            model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(10, activation='relu', input_shape=(history, 1)),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
        
        #Add more models here...
        
        return semiSupervised, history, model

# example of implementation
class mad:

    def __init__(self,method = Models.ONE_CLASS_SVM):
        self._semiSupervised = False
        self.method = method
        self.model = self.getModel()
    
    def getModel(self):
        self._semiSupervised, self._history, model = Models.getModel(self.method)
        return model

    def prepareForcastDataset(self,data,history = 5):
        datasets = {}
        for trace in data:                          #For each trace
            for i in range(len(trace)-history):   #For each moment in trace
                for j in range(len(trace[i])):      #For each feature of specific moment
                    if "f"+str(j) not in datasets:
                        #Create as many datasets as the features number
                        #We are planning to do forecast on each feature
                        datasets["f"+str(j)] = {"x":[],"y":[]}
                    #Based on the past histort moments, try to predict the next value of a feature
                    x_history = []
                    for k in range(i,i+history):
                        x_history.append(trace[k][j])
                    datasets["f"+str(j)]["x"].append(x_history)
                    datasets["f"+str(j)]["y"].append(trace[i+history][j])
                
        return datasets

    def fit(self,trainingdata,history = 5) -> None:
        x_train = trainingdata["train"]
        if self._semiSupervised:    #>> Semi-Supervised
            data = self.prepareForcastDataset(x_train)
            models = []
            for key in data: #For each feature
                models.append( #Train one model
                    self.getModel().fit(data[key]["x"],data[key]["y"])) 
            #We are training one model per feature to do forecast on its value
            #The idea is to try to reconstract the time series of each feature differently
            #So we can learn the normal series and find any outliers
            self.model = models
            self._history = history
        else:                       #>> Unsupervised
            allData = []
            for trace in x_train:
                for features in trace:
                    allData.append(features)
            self.model = self.model.fit(allData)

    def predict(self,records,threshold=0.5):
        scores = self.score(records)
        return [ 1 if a > threshold else 0 for a in scores ] # 0=Normal, 1=Anomaly

    #Idea what if we insert noice? Problem, this noice might mess up the interpretability...
    #def addNoice(self,series,labels, width = 5):
    #    newTrace = []
    #    index = randint(0,len(trace)) #Select a random index to insert the anomaly
    #    for i in range(0,index): #Split the series in the selected index
    #        newTrace.append(series[index])
    #    for i in range(0,width): #Insert anomaly
    #        newTrace.append([j+(1+random()) for j in series[i]])
    #    if index < len(trace): #Insert the rest of the series
    #        for i in range(index,len(trace)):
    #            newTrace.append(series[index])

    def score(self,records):
        #Ranges from 0 to 1, everything greater than 0.5 could be an anomaly
        if self._semiSupervised:
            
            #Find how off are we while tring to forcast the feature values based in what we trained
            scores = []
            for i in range(len(records)-self._history):   #For each time in series
                j = 0
                loss = 0
                for m in self.model:        #For each feature in specific time
                    x_history = []
                    for k in range(i,i+self._history):
                        x_history.append(records[k][j])
                    pred = m.predict([x_history])[0]
                    act = records[i+self._history][j]
                    loss += abs(pred - act)
                    j += 1
                scores.append(loss/len(self.model))
                i += 1
            
            #Normalize losses and return :
            min_score = min(scores)
            max_score = max(scores)
            return [(x - min_score) / (max_score - min_score) for x in scores]
        else:
            scores = self.model.decision_function(records)
            return [(-1*s+1)/2 for s in scores] #From 0 to 1, the greater the more anomalus

    def getscores(self,data):
        numpy_3d_data = data
        scores2d = []
        for period in numpy_3d_data:
            #for record in period:
            scoresforPeriod = self.score(period)
            scores2d.append(np.array(scoresforPeriod))
        # Remember the returned score must be in shape (#periods,#records)
        return np.array(scores2d)

    def getpreds(self,data,th=0.9):
        numpy_3d_data = data
        predictions2d = []
        for period in numpy_3d_data:
            #for record in period:
            pred = self.predict(period,th)
            predictions2d.append(np.array(pred))
        # Remember the returned predictions must be in shape (#periods,#records) with binary (1,0) values.
        return np.array(predictions2d)