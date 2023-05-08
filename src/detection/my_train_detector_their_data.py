import os
import argparse
import importlib
import itertools
import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME, MODELING_TEST_NAME, parsers,
    get_output_path, get_args_string, get_modeling_task_and_classes
)
from data.helpers import load_mixed_formats, load_datasets_data
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from metrics.evaluation import save_evaluation
from metrics.ad_evaluators import evaluation_classes
from detection.detector import Detector



parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument(
    '--evaluation_type', default='ad2',
    choices=['point', 'range', 'ad1', 'ad2', 'ad3', 'ad4'],
)
parser.add_argument(
    '--f_score_beta', default=1.0,
)
parser.add_argument(
    '--recall_alpha', default=0.0,
)
parser.add_argument(
    '--recall_omega', default='default',
)
parser.add_argument(
    '--recall_delta', default='flat',
)
parser.add_argument(
    '--recall_gamma', default='dup',
)
parser.add_argument(
    '--precision_omega', default='default',
)
parser.add_argument(
    '--precision_delta', default='flat',
)
parser.add_argument(
    '--precision_gamma', default='dup',
)
parser.add_argument(
    '--data', default='spark',
)






def getscores(data):
    randomscores=[0,0.1,0.2,0.1,0.2,0.3,0.4,0.3,0.1,0,0.1]
    size_rand_scores=len(randomscores)
    numpy_3d_data=data['test']
    scores=[]
    for period in numpy_3d_data:
        period_scores=[]
        counter = 0
        for record_in_period in period:
            period_scores.append(randomscores[counter%size_rand_scores])
            counter+=1
        scores.append(np.array(period_scores))
    return np.array(scores)

def getpreds(data,th):
    numpy_3d_data = data['test_scores']
    scores = []
    for period in numpy_3d_data:
        period_scores=[]
        for record_in_period in period:
            if record_in_period>=th:
                period_scores.append(1)
            else:
                period_scores.append(0)
        scores.append(np.array(period_scores))
    return np.array(scores)
if __name__ == '__main__':
    # parse and get command-line arguments
    argsold = parsers['train_detector'].parse_args()
    # set input and output paths
    DATA_INFO_PATH = get_output_path(argsold, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(argsold, 'build_features', 'data')
    MODEL_INPUT_PATH = get_output_path(argsold, 'train_model')
    SCORER_INPUT_PATH = get_output_path(argsold, 'train_scorer')
    OUTPUT_PATH = get_output_path(argsold, 'train_detector', 'model')
    COMPARISON_PATH = get_output_path(argsold, 'train_detector', 'comparison')

    # load the periods records, labels and information used to derive and evaluate anomaly predictions
    data_sets = [PIPELINE_TEST_NAME]
    # data contain:
    # test 3D data, (many periods , each period many records, each record many features
    # y_test 2D data  (for each period an array which indicate the anomaly type of records
    # info : for each period [filename,anomaly_type] -> [[filename,anomaly_type],...,[filename,anomaly_type]]
    data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, data_sets)

    data_combine_train_test=False
    if data_combine_train_test:
        data_sets2 = [PIPELINE_TRAIN_NAME]
        # data2 contain:
        # train 3D data, (many periods , each period many records, each record many features
        # y_train 2D data  (for each period an array which indicate the anomaly type of records
        # info : for each period [filename,anomaly_type] -> [[filename,anomaly_type],...,[filename,anomaly_type]]
        data2 = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, data_sets2)
        #print(data2)
        #exit(123)






        data["test"]=np.append(data2['train'],data["test"])
        data["y_test"]=np.append(data2['y_train'],data["y_test"])
        data2["train_info"].extend(data['test_info'])
        data["test_info"]=data2['train_info']



    data['test_scores']=getscores(data)
    th = 0.2
    data['test_preds'] = getpreds(data, th)

    args = parser.parse_args()


    spreadsheet_path = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myADoutput"
    OUTPUT_PATH = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myADoutput/scoringOutput"


    evaluator = evaluation_classes[args.evaluation_type](args)

    # this is equal to ad2_1.0
    evaluation_string = get_args_string(args, 'ad_evaluation')

    # unique configuration identifier serving as an index in the spreadsheet
    config_name = "unique_name_001"
    #(must be either "scoring", "detection" or "explanation")
    save_evaluation(
        'detection', data, evaluator, evaluation_string, config_name,
        spreadsheet_path,  used_data=args.data,method_path=OUTPUT_PATH
    )

