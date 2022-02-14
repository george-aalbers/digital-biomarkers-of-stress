'''
Description
---
This function builds one ML experiment from scratch. This function takes as input a list with study parameters for one experiment and runs the experiment by using other functions.

Input
---
param study_parameters: a list with study parameters

Output
---
The output of this script is a completed experiment.

'''

# Import 
from preprocessing import select_features_targets_ids, feature_extraction, aggregate_targets, drop_participants, drop_superfluous_columns
from train_models import train_models
import pandas as pd
import os

# Define function for doing one experiment
def do_experiment(study_parameters):

    # Only read data if we have not done feature extraction yet
    if "data.csv" not in os.listdir():
    
        log_data = pd.read_csv(study_parameters["log_data_path"], index_col = 0)
        esm_data = pd.read_csv(study_parameters["esm_data_path"], index_col = 0, low_memory = False, usecols = study_parameters["self_report_columns"])
        
        # Drop missingness in the targets
        esm_data.dropna(subset = study_parameters["non_aggregated_targets"], inplace = True)
    
        print("Currently doing feature extraction.")
        print("===")
        print("===")
        print("===")
    
        # Extract features
        feature_extraction(esm_data, log_data, study_parameters)

        print("Aggregating targets.")
        print("===")
        print("===")
        print("===")
        
        # Aggregate targets
        aggregate_targets(study_parameters)

        print("Dropping participants.")
        print("===")
        print("===")
        print("===")
        
        # Drop participants with < 10 observations
        drop_participants(study_parameters)
        
        # Remove superfluous columns
        drop_superfluous_columns(study_parameters)
   
    # Only select features/targets/ids if we have not written them to file yet
    if "X.csv" not in os.listdir():
    
        # Get X, y, and ids
        select_features_targets_ids(study_parameters)
  
    print("Training models.")
    print("===")
    print("===")
    print("===")
    
    # Train models
    train_models(study_parameters)

    print("Finished running the experiment.")
    print("===")
    print("===")
    print("===")