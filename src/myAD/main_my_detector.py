import helper
from mydetector import mad

if __name__ == '__main__':
    data=helper.get_test_data()
    datatrain=helper.get_train_data()

    model=mad()

    model.fit(datatrain["train"])

    data['test_scores']=model.getscores(data["test"])

    data['test_preds'] =model.getpreds(data["test"],th=0.95)


    helper.evaluation(data,experiment_name="try_0000",scores=True)

