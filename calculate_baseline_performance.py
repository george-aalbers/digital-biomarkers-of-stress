import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from preprocessing import within_person_center_idiographic

study_parameters = pd.read_json("study_parameters.json")

def mean_absolute_error_single_subject_nomothetic(df, study_parameters):
    
    # Select targets
    y_true = df[study_parameters["targets"]]
    
    # Center targets
    y_true = y_true - y_true.mean()
    
    # Get predictions from baseline model
    y_pred = np.repeat(0, y_true.shape[0])
    
    return mean_absolute_error(y_true, y_pred)

def mean_absolute_error_single_subject_idiographic(df, study_parameters):
    
    # Select targets
    y = df[study_parameters["targets"]]
    
    # Get size of test set
    test_size = np.round(y.shape[0] * study_parameters["time_series_test_size"], decimals = 0).astype(int)

    # Select train and test data
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    # Within-person center train and test data based on mean in the train set
    y_train, y_test = within_person_center_idiographic(y_train, y_test, study_parameters)

    # Get predictions of baseline model
    y_pred = np.repeat(0, y_test.shape[0])

    return mean_absolute_error(y_test, y_pred)

def calculate_baseline_performance():
    
    study_parameters = pd.read_json("study_parameters.json")
    
    for index, experiment in study_parameters.iterrows():
        
        # Read participant IDs and targets
        cols = [experiment["id_variable"], experiment["targets"]]
        
        data = pd.read_csv("data.csv", usecols = cols)
    
        if experiment["experiment_type"] == "nomothetic":
            
            baseline_performance_nomothetic = data.groupby("id").apply(mean_absolute_error_single_subject_nomothetic, experiment)
            baseline_performance_nomothetic = baseline_performance_nomothetic.reset_index()
            baseline_performance_nomothetic.index = np.repeat("nomothetic", baseline_performance_nomothetic.shape[0])
            baseline_performance_nomothetic.reset_index(inplace = True)
        
        elif experiment["experiment_type"] == "idiographic":
            
            baseline_performance_idiographic = data.groupby("id").apply(mean_absolute_error_single_subject_idiographic, experiment)
            baseline_performance_idiographic = baseline_performance_idiographic.reset_index()
            baseline_performance_idiographic.index = np.repeat("idiographic", baseline_performance_idiographic.shape[0])
            baseline_performance_idiographic.reset_index(inplace = True)            
            
    baseline_performance_multiple_experiments = pd.concat([baseline_performance_nomothetic, baseline_performance_idiographic], axis = 0)
    baseline_performance_multiple_experiments.columns = ["experiment_type", "id", "mae"] 
    
    # Write to file
    directory = os.getcwd()
    newpath = 'baseline'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    baseline_performance_multiple_experiments.to_csv(directory + "/baseline/baseline_performance.csv")