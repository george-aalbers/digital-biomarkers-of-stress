'''
Description
---
This function generates a .json file containing instructions for the machine learning pipeline. 

'''

import pandas as pd
import numpy as np
import os

def study_parameters():
    
    # Root folder for the directory structure
    root_folder = os.getcwd()     

    # Number of experiments
    n_experiments = 6
    
    # All required columns of the self-report data
    self_report_columns = {'id', 'Response Time_ESM_day', 'I feel relaxed', 'I feel stressed (tense restless nervous or anxious)'}
 
    # Smartphone application categories to select in feature extraction
    categories = {'Calling', 'Camera', 'Dating', 'Email', 'Game', 'Video', 'Tracker', 'Social Networks', 'Music & Audio', 'Exercise', 
                  'Work', 'Food & Drink', 'Gallery', 'Productivity', 'Browser', 'Messenger', 'Shared Transportation','Weather'}
    
    # Feature names
    features = {# Smartphone application usage features 
                'Work (duration)',
                'Messenger (duration)',
                'Weather (duration)',
                'Video (duration)',
                'Tracker (duration)',
                'Social Networks (duration)',
                'Dating (duration)',
                'Gallery (duration)',
                'Food & Drink (duration)',
                'Productivity (duration)',
                'Calling (duration)',
                'Exercise (duration)',
                'Game (duration)',
                'Music & Audio (duration)',
                'Shared Transportation (duration)',
                'Browser (duration)',
                'Camera (duration)',
                'Email (duration)',
                'Work (frequency)',
                'Messenger (frequency)',
                'Weather (frequency)',
                'Video (frequency)',
                'Tracker (frequency)',
                'Social Networks (frequency)',
                'Dating (frequency)',
                'Gallery (frequency)',
                'Food & Drink (frequency)',
                'Productivity (frequency)',
                'Calling (frequency)',
                'Exercise (frequency)',
                'Game (frequency)',
                'Music & Audio (frequency)',
                'Shared Transportation (frequency)',
                'Browser (frequency)',
                'Camera (frequency)',
                'Email (frequency)',
        
                # Sleep proxies
                "Sleep onset", 
                "Sleep duration", 
                
                # Temporal features
                "Hour of day", 
                "Day of week", 
                "Day of month", 
                "COVID-19"}
    
    # Names of the original target items we work with
    non_aggregated_targets = {'I feel relaxed', 'I feel stressed (tense restless nervous or anxious)'}
    
    # Name of the target after aggregating the original items
    targets = "stress"
    
    # Model types in this study 
    models = ["lasso", "svr", "rf", "lasso", "svr", "rf"]
    
    # Hyperparameters of the models we train in the study
    lasso_parameters_nomothetic = {"alpha": np.arange(0.0001,0.005,0.0001)}
    
    svm_parameters_nomothetic = {"kernel": ["rbf"],
                      "gamma": ["scale"],
                      "C": [0.1, 0.5, 1],
                      "epsilon": [0.7, 0.75, 0.8]}
    
    rf_parameters_nomothetic = {'n_estimators': [200, 300, 400, 500],
                     'min_samples_leaf': [0.0001, 0.0002, 0.0005], 
                     'min_samples_split': [0.0001, 0.0005, 0.001],
                     'min_weight_fraction_leaf': [0.0001, 0.0005, 0.001],
                     'max_features': [18, 19, 20],
                     'max_leaf_nodes' : [200, 400],
                     'max_depth' : [30, 33, 35]
                     }

    lasso_parameters_idiographic = {"alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    
    svm_parameters_idiographic = {"kernel": ["rbf"],
                                  "gamma": ["scale","auto"],
                                  "C": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000],
                                  "epsilon": [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]}
    
    rf_parameters_idiographic = {'n_estimators': [10, 20, 50, 100, 200, 500, 1000, 2000],
                                 'min_samples_leaf': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5], 
                                 'min_samples_split': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                                 'min_weight_fraction_leaf': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                                 'max_features': ["auto", "sqrt", "log2"],
                                 'max_leaf_nodes' : [2, 5, 10, 20, 40, 50, 100],
                                 'max_depth' : [1, 5, 10, 20, 40, 50, 100]
                                 }
    
    model_parameters = [lasso_parameters_nomothetic, svm_parameters_nomothetic, rf_parameters_nomothetic, 
                        lasso_parameters_idiographic, svm_parameters_idiographic, rf_parameters_idiographic]
    
    # Dataframe containing instructions for the study
    study_parameters = pd.DataFrame({"esm_data_path":             np.repeat('/home/haalbers/dissertation/experience-sampling-clean.csv', n_experiments),
                                     "log_data_path":             np.repeat("/home/haalbers/dissertation/mobiledna-clean.csv", n_experiments),
                                     "sleep_features_path":       np.repeat("/home/haalbers/dissertation/sleep-features.csv", n_experiments),
                                     "baseline_path":             np.repeat(root_folder + "/baseline/baseline_performance.csv", n_experiments),
                                     "data_output_path":          (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "data/").values,
                                     "model_output_path":         (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "models/").values,
                                     "results_output_path":       (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "results/").values,
                                     "explanations_output_path":  (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "explanations/").values,
                                     "visualizations_output_path":(root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "visualizations/").values,
                                     "data_samples_output_path":  (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "samples/").values,
                                     "markdown_path":             np.repeat(root_folder + "/markdown/", n_experiments),
                                     "id_variable":               np.repeat("id", n_experiments),
                                     "categories":                np.tile(categories, n_experiments),
                                     "features":                  np.tile(features, n_experiments),
                                     "non_aggregated_targets":    np.tile(non_aggregated_targets, n_experiments),
                                     "targets":                   np.tile(targets, n_experiments),
                                     "self_report_columns":       np.tile(self_report_columns, n_experiments),
                                     "experiment":                range(1, n_experiments + 1),
                                     "experiment_type":           ["nomothetic", "nomothetic", "nomothetic", "idiographic", "idiographic", "idiographic"],
                                     "window_size":               np.repeat(60, n_experiments),
                                     "prediction_task":           np.repeat("regression", n_experiments),
                                     "cross_validation_type":     ["grid","grid","grid","random","random","random"],
                                     "outer_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "inner_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "time_series_k_splits":      np.repeat(1, n_experiments),
                                     "time_series_test_size":     np.repeat(0.2, n_experiments),
                                     "n_jobs":                    np.repeat(124, n_experiments),
                                     "model_type":                models,
                                     "model_parameters":          model_parameters}, 
                                    index = np.arange(n_experiments))
    
    # Write this dataframe to .json
    study_parameters.to_json("study_parameters.json")
    
study_parameters()