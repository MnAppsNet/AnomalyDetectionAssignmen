import helper
from mydetector import mad, Models

# Config :
TEST_CASE_NAME = "OneClassSVM_0001"
MODEL = Models.RNN
SCORES = True

if __name__ == '__main__':
    data=helper.get_test_data()
    datatrain=helper.get_train_data()

    print(f"Loading model '{MODEL}'...")
    model=mad(MODEL)

    print("Training model...")
    model.fit(datatrain)

    print("Model trained, generating scores and predictions...")
    if SCORES:
        data['test_scores']=model.getscores(data["test"])
    else:
        data['test_preds']=model.getpreds(data["test"],th=0.85)

    print("Evaluating model...")
    helper.evaluation(data,experiment_name=TEST_CASE_NAME,scores=SCORES)