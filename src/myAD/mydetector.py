import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models import Models
from random import choice, randint, random

# example of implementation
class mad:

    def __init__(self,method = Models.ONE_CLASS_SVM, multiModel = False, multiModelType = Models.MultiModelTypes.CrossModels):
        self._method = method
        self._multiModel = multiModel
        self._multiModelType = multiModelType
        self._model = self.getModel()
        pass
    
    def getModel(self):
        model, self._unsupervised = Models.getModel(self._method)
        return model

    def fit(self,trainingdata,data) -> None:
        models = []
        if self._unsupervised:  #Unsupervised method, give it all the test data and let it do its thing...
            x = data["test"]
        else:                   #Semi-Superviced, let's try to teach it what normal data looks like...
            x = trainingdata["train"]
        
        if self._multiModel:
            #Fit one model per trace
            for trace in x:
                trace =  MinMaxScaler().fit_transform(trace)
                models.append(self.getModel().fit(trace))
        else:
            #Combine traces into a big series and train the model with it
            allData = []
            for trace in x:
                allData.extend(trace)
            allData = MinMaxScaler().fit_transform(allData)
            models = self.getModel().fit(allData)

        self.model = models


    def predict(self,records,threshold=0.5,traceID=-1):
        scores = self.score(records,traceID)
        return [ 1 if a > threshold else 0 for a in scores ] # 0=Normal, 1=Anomaly

    def score(self,records,traceID=-1):
        #Ranges from 0 to 1, the more possitive the more outlier
        scores = []
        final_scores = []
        if self._multiModel and self._multiModelType == Models.MultiModelTypes.CrossModels:
            for model in self.model:
                scores.append([(-1*s+1)/2 for s in model.decision_function(records)])
            for score in np.array(scores).T:
                final_scores.append(sum(score)/len(score))
        elif self._multiModel and self._multiModelType == Models.MultiModelTypes.SingleModel and traceID != -1:
            final_scores = self.model[traceID].decision_function(records)
            if min(final_scores) < 0:
                final_scores = [(-1*s+1)/2 for s in final_scores]
        else:
            final_scores = self.model.decision_function(records)
            if min(final_scores) < 0:
                final_scores = [(-1*s+1)/2 for s in final_scores]

        return final_scores

    def getscores(self,data):
        numpy_3d_data = data
        scores2d = []
        i = 0
        for period in numpy_3d_data:
            period = MinMaxScaler().fit_transform(period)
            #for record in period:
            scoresforPeriod = self.score(period,i)
            scores2d.append(np.array(scoresforPeriod))
            i += 1
        # Remember the returned score must be in shape (#periods,#records)
        return np.array(scores2d)

    def getpreds(self,data,th=0.9):
        numpy_3d_data = data
        predictions2d = []
        i = 0
        for period in numpy_3d_data:
            #for record in period:
            pred = self.predict(period,th,i)
            predictions2d.append(np.array(pred))
            i += 1
        # Remember the returned predictions must be in shape (#periods,#records) with binary (1,0) values.
        return np.array(predictions2d)