
import numpy as np
from sklearn.svm import OneClassSVM

# example of implementation
class mad:

    def __init__(self):
        pass

    def fit(self,trainingdata):
        # use data to model normality
        return
    def predict(self,record,threshold=0):
        return 1

    def score(self,record):
        return 1


    def getscores(self,data):

        numpy_3d_data = data
        scores2d = []
        for period in numpy_3d_data:
            scoresforPeriod=[]
            for record in period:
                scoresforPeriod.append(self.score(record))
            scores2d.append(np.array(scoresforPeriod))
        # Remember the returned score must be in shape (#periods,#records)
        return np.array(scores2d)

    def getpreds(self,data,th=0.8):
        numpy_3d_data = data
        predictions2d = []
        for period in numpy_3d_data:
            predictionsforPeriod = []
            for record in period:
                predictionsforPeriod.append(self.predict(record))
            predictions2d.append(np.array(predictionsforPeriod))
        # Remember the returned predictions must be in shape (#periods,#records) with binary (1,0) values.
        return np.array(predictions2d)