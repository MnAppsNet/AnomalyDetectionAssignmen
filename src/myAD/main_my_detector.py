import helper, numpy as np, sys
from mydetector import mad, Models
from models import Models

def Get_Timestep_Differences(x):
    data = []
    for trace in x:
        new_trace = []
        for i in range(len(trace) - 1):
            new_trace.append(trace[i+1] - trace[i])
        new_trace.append([0 for i in range(19)]) #19 Features
        data.append(np.array(new_trace))
    return np.array(data)


def main( ITERATION, MODEL, SCORES, MULTIPLE_MODELS, TRAIN_ON_CHANGES, 
            MODELS, WINDOW, DATA_SMOOTHING_WINDOW, SCORE_SMOOTHING_WINDOW, MULTI_MODEL_TYPE):
    data=helper.get_test_data()
    datatrain=helper.get_train_data()

    if TRAIN_ON_CHANGES:
        datatrain["train"] = Get_Timestep_Differences(datatrain["train"])
        data["test"] = Get_Timestep_Differences(data["test"])
        

    ITERATION = ( 
        ("MULTI_" if MULTIPLE_MODELS==True else "SINGLE_") + 
        ("" if MULTIPLE_MODELS==False else f"{MODELS}_") +
        ("" if TRAIN_ON_CHANGES==False else f"DIFF_") +
        ("" if MODEL != Models.AUTO_ENCODER else f"WIN_{WINDOW}_") +
        ITERATION )

    print(f"Loading model...")
    model=mad(MODEL,MULTIPLE_MODELS,MULTI_MODEL_TYPE,MODELS,WINDOW,DATA_SMOOTHING_WINDOW,SCORE_SMOOTHING_WINDOW)

    print("Training model...")
    model.fit(datatrain,data)

    print("Model trained, generating scores and predictions...")
    if SCORES:
        data['test_scores']=model.getscores(data["test"])
    else:
        data['test_preds']=model.getpreds(data["test"],th=0.6)

    print("Evaluating model...")
    helper.evaluation(data,experiment_name=f"{MODEL}_{ITERATION}",scores=SCORES)

if __name__ == '__main__':

    if len(sys.argv) == 11:
        ITERATION = sys.argv[1]
        MODEL = sys.argv[2]
        SCORES = sys.argv[3]
        MULTIPLE_MODELS = sys.argv[4]
        TRAIN_ON_CHANGES = sys.argv[5]
        MODELS = sys.argv[6]
        WINDOW = sys.argv[7]
        DATA_SMOOTHING_WINDOW = sys.argv[8]
        SCORE_SMOOTHING_WINDOW =  sys.argv[9]
        MULTI_MODEL_TYPE = sys.argv[10]
    else:
        ITERATION = "W"
        MODEL = Models.AUTO_ENCODER
        SCORES = True
        MULTIPLE_MODELS = False
        TRAIN_ON_CHANGES = True #Train on the changes between two timesteps instead of the actual values of the timestep
        MODELS = 5 #Relevant for Multiple Models, If set to -1 then we have as many models as the traces
        WINDOW = 1 #Relevant for LSTM and AutoEncoder
        DATA_SMOOTHING_WINDOW = 9  #Use 0 to not smooth
        SCORE_SMOOTHING_WINDOW = 5 #Use 0 to not smooth
        MULTI_MODEL_TYPE = Models.MultiModelTypes.CrossModels


    main(   ITERATION = ITERATION,
        MODEL = MODEL,
        SCORES = SCORES,
        MULTIPLE_MODELS = MULTIPLE_MODELS,
        TRAIN_ON_CHANGES = TRAIN_ON_CHANGES, 
        MODELS = MODELS, 
        WINDOW = WINDOW, 
        DATA_SMOOTHING_WINDOW = DATA_SMOOTHING_WINDOW, 
        SCORE_SMOOTHING_WINDOW = SCORE_SMOOTHING_WINDOW, 
        MULTI_MODEL_TYPE = MULTI_MODEL_TYPE)
