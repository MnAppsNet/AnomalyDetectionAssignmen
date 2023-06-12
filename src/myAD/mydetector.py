import numpy as np
from sklearn.model_selection import train_test_split
from models import Models
from random import choice, randint, random
from scipy.signal import savgol_filter

# example of implementation
class mad:

    def __init__(self,method = Models.ONE_CLASS_SVM, multiModel = False, multiModelType = Models.MultiModelTypes.CrossModels, model_num = -1, window = 10, data_smooth_window = 9, score_smooth_window = 0):
        self._method = method
        self._multiModel = multiModel
        self._multiModelType = multiModelType
        self.model_num = model_num
        self.window = window
        self.data_smooth_window = data_smooth_window
        self.score_smooth_window = score_smooth_window
        self._model = self.getModel()
        pass
    
    def getModel(self):
        model, self._unsupervised = Models.getModel(self._method,self.window)
        return model

    def scale_trace(self,trace):
        trace = np.array(trace)
        #min_feature_value = min(trace.reshape(-1,1))
        #max_feature_value = max(trace.reshape(-1,1))
        #scaled = (trace - min_feature_value) / (max_feature_value-min_feature_value)
        #Smooth feature values :
        if self.data_smooth_window > 0:
            smoothed = savgol_filter(trace,self.data_smooth_window,3)
        else:
            smoothed = trace
        return smoothed

    def split_traces_in_groups(self,traces,groups=-1):
        if groups == - 1: groups = self.model_num
        allData = []
        splittedData = []
        for trace in traces:
            allData.extend(trace)
        for i in range(0,groups):
            fromIndex = int(i/groups*len(allData))
            toIndex = int((i+1)/groups*len(allData))
            if toIndex >= len(allData):
                toIndex = len(allData) - 1
            splittedData.append(allData[fromIndex:toIndex])
        return np.array(splittedData)
            

    def fit(self,trainingdata,data) -> None:
        models = []
        if self._unsupervised:  #Unsupervised method, give it all the test data and let it do its thing...
            x = data["test"]
        else:                   #Semi-Superviced, let's try to teach it what normal data looks like...
            x = trainingdata["train"]
        
        if self._multiModel:
            #Fit one model per trace
            if self.model_num == -1:
                for trace in x:
                    #Scale features :
                    scaled_trace = self.scale_trace(trace)
                    models.append(self.getModel().fit(scaled_trace))
            else:
                x = self.split_traces_in_groups(x)
                for i in range(0,self.model_num):
                    scaled_trace = self.scale_trace(x[i])
                    models.append(self.getModel().fit(scaled_trace))
        else:
            #Combine traces into a big series and train the model with it
            allData = []
            for trace in x:
                allData.extend(trace)
            #Scale features :
            scaled_data = self.scale_trace(allData)
            models = self.getModel().fit(scaled_data)

        self.model = models


    def predict(self,records,threshold=0.5,traceID=-1):
        scores = self.score(records,traceID)
        return [ 1 if a > threshold else 0 for a in scores ] # 0=Normal, 1=Anomaly

    def scale_scores(self,scores):
        if min(scores) < 0:
            #If outliers are marked as negative, inverse the logic, we need them to be
            #as close to 1 as possible while normal instances close to 0
            scores = [-1*s for s in scores]
            scores = (scores - min(scores))/(max(scores)-min(scores))
        if max(scores) > 1:
            scores = (scores - min(scores))/(max(scores)-min(scores))
        if self.score_smooth_window > 0:
            smoothed = savgol_filter(scores,self.score_smooth_window,3)
        else:
            smoothed = scores
        return smoothed

    def score(self,records,traceID=-1):
        #Ranges from 0 to 1, the more possitive the more outlier
        scores = []
        final_scores = []
        if self._multiModel and self._multiModelType == Models.MultiModelTypes.CrossModels:
            scores = []
            for model in self.model:
                scores.append(self.scale_scores(model.decision_function(records)))
            for score in np.array(scores).T:
                final_scores.append(sum(score)/len(score))
        elif self._multiModel and self._multiModelType == Models.MultiModelTypes.SingleModel and traceID != -1:
            final_scores = self.scale_scores(self.model[traceID].decision_function(records))
        else:
            final_scores = self.scale_scores(self.model.decision_function(records))

        return final_scores

    def getscores(self,data):
        numpy_3d_data = data
        scores2d = []
        original_trace_lengths = []
        i = 0

        if self._multiModelType == Models.MultiModelTypes.SingleModel:
            #Split the timestps of all traces into 'self.model_num' groups
            for t in numpy_3d_data:
                original_trace_lengths.append(len(t))
            numpy_3d_data = self.split_traces_in_groups(numpy_3d_data)
            
        for period in numpy_3d_data:
            #for record in period:
            scaled_period = self.scale_trace(period)
            scoresforPeriod = self.score(scaled_period,i)
            #scoresforPeriod = savgol_filter(scoresforPeriod, 9, 2)
            scores2d.append(np.array(scoresforPeriod))
            i += 1
        
        if self._multiModelType == Models.MultiModelTypes.SingleModel:
            #Format the scores in the same way as they traces were given
            allScores = []
            for s in scores2d:
                    allScores.extend(s)
            scores = []
            j = 0
            for l in original_trace_lengths:
                scores.append(np.array(allScores[j:j+l-1]))
                j += l
            scores2d = scores

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