import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class Method:
    ONE_CLASS_SVM = "OneClassSVM"
    ISOLATION_FOREST = "IsolationForest"

# example of implementation
class mad:

    def __init__(self,method = Method.ONE_CLASS_SVM):
        if method == Method.ONE_CLASS_SVM:
            self.model = OneClassSVM(gamma='auto')
        elif method == Method.ISOLATION_FOREST:
            self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42)

    def fit(self,trainingdata) -> None:
        allData = []
        for trace in trainingdata:
            for data in trace:
                allData.append(data)
        self.model = self.model.fit(allData)

    def predict(self,records,threshold=0):
        return [ 1 if a == -1 else 0 for a in self.model.predict(records) ] # 0=Normal, 1=Anomaly

    def score(self,records):
        return self.model.decision_function(records)


    def getscores(self,data):
        numpy_3d_data = data
        scores2d = []
        for period in numpy_3d_data:
            #for record in period:
            scoresforPeriod = self.score(period)
            scores2d.append(np.array(scoresforPeriod))
        # Remember the returned score must be in shape (#periods,#records)
        return np.array(scores2d)

    def getpreds(self,data,th=0.8):
        numpy_3d_data = data
        predictions2d = []
        for period in numpy_3d_data:
            #for record in period:
            pred = self.predict(period)
            predictions2d.append(np.array(pred))
        # Remember the returned predictions must be in shape (#periods,#records) with binary (1,0) values.
        return np.array(predictions2d)