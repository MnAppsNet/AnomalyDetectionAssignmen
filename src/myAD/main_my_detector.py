import helper
from mydetector import mad, Method

# Config :
TEST_CASE_NAME = "OneClassSVM_002"
MODEL = Method.ONE_CLASS_SVM


if __name__ == '__main__':
    data=helper.get_test_data()
    datatrain=helper.get_train_data()

    print(f"Loading model '{MODEL}'...")
    model=mad(MODEL)

    print("Training model...")
    model.fit(datatrain["train"])

    print("Model trained, generating scores and predictions...")
    data['test_scores']=model.getscores(data["test"])
    data['test_preds']=model.getpreds(data["test"],th=0.95)

    print("Evaluating model...")
    helper.evaluation(data,experiment_name=TEST_CASE_NAME,scores=True)