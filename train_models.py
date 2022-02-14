'''
Description
---
This function executes a nested cross-validation.

Input
---
param experiment_type: type of experiment to run (idiographic or nomothetic)
param test_size: number of individuals in the test set.
param data_output_path: location of non-splitted data
param features: features to include in X_train, X_test
param targets: targets to include in y_train, y_test

Output
---
The output of this script is four files: X_train.csv, X_test.csv, y_train.csv, and y_test.csv. 

'''

import pandas as pd
import os
from build_model import build_model
from train_model import train_model

def train_models(study_parameters):
    
    root = os.getcwd()
    
    model = build_model(study_parameters)
    
    if study_parameters["experiment_type"] == "nomothetic":
    
        for loop_id in range(1,6,1):

            X_path = study_parameters["data_output_path"] + "X_train_" + str(loop_id) + ".csv"
            y_path = study_parameters["data_output_path"] + "y_train_" + str(loop_id) + ".csv"

            X = pd.read_csv(X_path, index_col = 0)
            y = pd.read_csv(y_path, index_col = 0)

            train_model(model, X, y, study_parameters, loop_id)
            
    elif study_parameters["experiment_type"] == "idiographic":
        
        participant_list = pd.read_csv("data.csv").id.unique().tolist()
        
        for loop_id in participant_list:

            X_path = study_parameters["data_output_path"] + "X_train_" + str(loop_id) + ".csv"
            y_path = study_parameters["data_output_path"] + "y_train_" + str(loop_id) + ".csv"

            X = pd.read_csv(X_path, index_col = 0)
            y = pd.read_csv(y_path, index_col = 0)

            train_model(model, X, y, study_parameters, loop_id)