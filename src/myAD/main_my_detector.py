import helper
from mydetector import mad, Models
from models import Models

# Config :
ITERATION = "0001"
MODEL = Models.ISOLATION_FOREST
SCORES = True
MULTIPLE_MODELS = True
MULTI_MODEL_TYPE = Models.MultiModelTypes.CrossModels

if __name__ == '__main__':
    data=helper.get_test_data()
    datatrain=helper.get_train_data()

    ITERATION = ( 
        ("MULTI_" if MULTIPLE_MODELS==True else "SINGLE_") + 
        ("" if MULTIPLE_MODELS==False else f"{MULTI_MODEL_TYPE}_") +
        ITERATION )

    print(f"Loading model...")
    model=mad(MODEL,MULTIPLE_MODELS,MULTI_MODEL_TYPE)

    print("Training model...")
    model.fit(datatrain,data)

    print("Model trained, generating scores and predictions...")
    if SCORES:
        data['test_scores']=model.getscores(data["test"])
    else:
        data['test_preds']=model.getpreds(data["test"],th=0.6)

    print("Evaluating model...")
    helper.evaluation(data,experiment_name=f"{MODEL}_{ITERATION}",scores=SCORES)