import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# example of implementation
class mad:

    def __init__(self):
        self.model = None
        pass

    def fit(self,trainingdata) -> None:
        allData = []
        for trace in trainingdata:
            for data in trace:
                allData.append(data)

        self.model = OneClassSVM(gamma='auto').fit(allData)

    def predict(self,records,threshold=0):
        return self.model.predict(records)

    def score(self,records):
        return self.model.score_samples(records)


    def getscores(self,data):
        numpy_3d_data = data
        scores2d = []
        for period in numpy_3d_data:
            scoresforPeriod=[]
            for record in period:
                scoresforPeriod.append(self.score([record])[0]) #We only get the first one because we only ask for one score
            scores2d.append(np.array(scoresforPeriod))
        # Remember the returned score must be in shape (#periods,#records)
        return np.array(scores2d)

    def getpreds(self,data,th=0.8):
        numpy_3d_data = data
        predictions2d = []
        for period in numpy_3d_data:
            predictionsforPeriod = []
            for record in period:
                predictionsforPeriod.append(self.predict([record])[0]) #We only get the first one because we only ask for one prediction
            predictions2d.append(np.array(predictionsforPeriod))
        # Remember the returned predictions must be in shape (#periods,#records) with binary (1,0) values.
        return np.array(predictions2d)