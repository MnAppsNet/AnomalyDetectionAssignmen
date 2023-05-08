import os
import argparse
import importlib
import itertools

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
import numpy as np

# 'evaluation_type': 'ad2',
#     'recall_alpha': 0.0,
#     'recall_omega': 'default',
#     'recall_delta': 'flat',
#     'recall_gamma': 'dup',
#     'precision_omega': 'default',
#     'precision_delta': 'flat',
#     'precision_gamma': 'dup',
#     'f_score_beta': 1.0,
# evaluation_type can be :
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

if __name__ == '__main__':
    args = parser.parse_args()





    spreadsheet_path="/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/pathFormyDetectorEvaluation"
    OUTPUT_PATH="/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/pathFormyDetectorEvaluation/scoringOutput"
    data_dict={
        #binary 0,1 or 0 , 1,2,3,4 for multiple types.
            "y_test":np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0]]),
        # [[filename,type]]
        "test_info":[["onefile","type_0"]],
        # binary 0,1
        "test_scores":np.array([[0,0,0.1,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0.2,0.3,0.4,0.5,0,0,0,0,0,0,0,0,0.1,0.1,0.7,0,0,0,0]]),
        "test_preds":np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
    }


    evaluator = evaluation_classes[args.evaluation_type](args)

    # this is equal to ad2_1.0
    evaluation_string = get_args_string(args, 'ad_evaluation')

    #unique configuration identifier serving as an index in the spreadsheet
    config_name="unique_name_1234"

    save_evaluation(
        'scoring', data_dict, evaluator, evaluation_string, config_name,
        spreadsheet_path,  method_path=OUTPUT_PATH
    )