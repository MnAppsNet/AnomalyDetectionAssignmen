import os
import argparse
from pathlib import Path

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import ( parsers, get_args_string )
from data.helpers import load_mixed_formats, load_datasets_data
from metrics.evaluation import save_evaluation
from metrics.ad_evaluators import evaluation_classes
from mydetector import mad


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

#
# data: dictionary with 'test', 'y_test', 'test_info' and 'test_scores' in case of scores equal True or
#     'test_preds' in case of scores equal False
# ev_type: the level of the evaluation one of the ['ad1', 'ad2', 'ad3', 'ad4']
# experiment_name: the name under which will be stored the results in the csv file
# scores: indicates the valuation of scores or predictions.
def evaluationSingleType(data,ev_type,experiment_name="try_1",scores=False):
    args = parser.parse_args()
    args.evaluation_type=ev_type
    spreadsheet_path = "myADoutput"
    OUTPUT_PATH = "myADoutput/scoringOutput"


    evaluator = evaluation_classes[ev_type](args)

    # this is equal to ad2_1.0
    evaluation_string = ev_type+"_"+str(args.f_score_beta)

    # unique configuration identifier serving as an index in the spreadsheet
    config_name = experiment_name
    # (must be either "scoring", "detection" or "explanation")

    if scores:
        save_evaluation(
            'scoring', data, evaluator, evaluation_string, config_name,
            spreadsheet_path, used_data="spark", method_path=OUTPUT_PATH
        )
    else:
        #for period in range(len(data['test_preds'])):
        #    data['test_preds'][period][0] = False
        save_evaluation(
            'detection', data, evaluator, evaluation_string, config_name,
            spreadsheet_path, used_data="spark", method_path=OUTPUT_PATH
        )
# data: dictionary with 'test', 'y_test', 'test_info' and 'test_scores' in case of scores equal True or
#     'test_preds' in case of scores equal False
# ev_type: the level of the evaluation one of the ['ad1', 'ad2', 'ad3', 'ad4']
# experiment_name: the name under which will be stored the results in the csv file
# scores: indicates the valuation of scores or predictions.
def evaluation(data,experiment_name="try_1",scores=False):
    for ev_type in ['ad1', 'ad2', 'ad3', 'ad4']:
        evaluationSingleType(data, ev_type, experiment_name=experiment_name, scores=scores)


# load the periods records, labels and information used to derive and evaluate anomaly predictions
# data contain:
# test 3D data, (many periods , each period many records, each record many features
# y_test 2D data  (for each period an array which indicate the anomaly type of records
# info : for each period [filename,anomaly_type] -> [[filename,anomaly_type],...,[filename,anomaly_type]]
def get_test_data():
    dataFolder = str(Path(__file__).parent)+"/preprossedData/"
    data = load_datasets_data(dataFolder, dataFolder, ["test"])
    return data
def get_train_data():
    dataFolder = str(Path(__file__).parent)+"/preprossedData/"
    datatrain = load_datasets_data(dataFolder, dataFolder, ["train"])
    return datatrain


