'''
Description
---
This function conducts an entire ML study from scratch. It takes as input the parameter specification for the entire study as well as path strings for 
the location of log and experience sampling data. It then conducts a study, which entails outputting an entire directory that contains per experiment:
- data - preprocessed and split into X_train, X_test, y_train, y_test
- models - trained models saved as .pkl files with model id signifying fold or participant ID
- explanations - SHAP values that form the basis of model explanation in the article
- samples - data samples used to calculate SHAP values
- visualizations - beeswarm plots visualising SHAP values against feature values
- results - person-specific Spearman rhos and mean absolute errors of y_pred versus y_test
'''
# Import required modules
import pandas as pd
import os
from study_parameters import study_parameters
from create_directory import create_directory
from preprocessing import feature_extraction, aggregate_targets, drop_participants, drop_superfluous_columns, select_features_targets_ids, split
from spearman_rho import rho_nomothetic_models, rho_idiographic_models
from mae import mae_nomothetic_models, mae_idiographic_models
from explain_model import feature_importance_nomothetic_models, feature_importance_idiographic_models, correlations_between_features_and_shapley_values_multiple_participants
from train_models import train_models
from calculate_baseline_performance import calculate_baseline_performance
from prepare_for_markdown import get_inputs
import time

# Specify the study parameters
print("Specifying the study parameters")
study_parameters()

# Read study specification
print("Reading study specification")
study_parameters = pd.read_json("study_parameters.json")
    
# Create directory for number of experiments
print("Creating folder structure")
create_directory(study_parameters.shape[0])

# Only read data if we have not done feature extraction yet
if "data.csv" not in os.listdir():

    print("Reading the raw data")
    log_data = pd.read_csv(study_parameters["log_data_path"][0], index_col = 0)
    esm_data = pd.read_csv(study_parameters["esm_data_path"][0], index_col = 0, low_memory = False, usecols = study_parameters["self_report_columns"][0])

    # Drop missingness in the targets
    print("Dropping missingness in the targets")
    esm_data.dropna(subset = study_parameters["non_aggregated_targets"][0], inplace = True)

    # Extract features
    print("Currently doing feature extraction.")
    feature_extraction(esm_data, log_data, study_parameters.iloc[0,:])

    # Aggregate targets
    print("Aggregating targets.")
    aggregate_targets(study_parameters.iloc[0,:])

    # Drop participants with < 6 observations
    print("Dropping participants.")
    drop_participants(study_parameters.iloc[0,:])

    # Remove superfluous columns
    print("Dropping columns.")
    drop_superfluous_columns(study_parameters.iloc[0,:])

# Select features/targets/ids 
if "X.csv" not in os.listdir():
    print("Selecting features, targets, and ids")
    select_features_targets_ids(study_parameters.iloc[0,:])

# Split data for all models
if "X_test_1.csv" not in os.listdir(study_parameters["data_output_path"][0]):
    print("Splitting data")
    for experiment in study_parameters.iterrows():
        print("===")
        print("===")
        print("===")
        print("Experiment #", experiment[1]["experiment"], sep = "")
        print("===")
        print("===")
        print("===")
        split(experiment[1])

# Train all models
if "lasso_1.pkl" not in os.listdir(study_parameters["model_output_path"][0]):
    print("Training models")
    for experiment in study_parameters.iterrows():
        print("===")
        print("===")
        print("===")
        print("Experiment #", experiment[1]["experiment"], sep = "")
        print("===")
        print("===")
        print("===")
        train_models(experiment[1])

# Evaluate all models
if "MAE_per_subject.csv" not in os.listdir(study_parameters["results_output_path"][0]):
    print("Calculating correlations between predictions and self-reports")
    rho_nomothetic_models()
    rho_idiographic_models()

    print("Calculating mean absolute error")
    mae_nomothetic_models()
    mae_idiographic_models()

# Calculate SHAP values for multiple models
print("Calculating SHAP values")
os.popen('sh /home/haalbers/fatigue-ml/shap.sh')
os.popen('sh /home/haalbers/fatigue-ml/shap-idiographic.sh')

# Calculate feature importance
print("Calculating feature importance")
feature_importance_nomothetic_models() 
feature_importance_idiographic_models()

print("Calculating correlations between SHAP values and features")
correlations_between_features_and_shapley_values_multiple_participants()

# Calculate the baseline
print("Calculating baseline performance")
calculate_baseline_performance()

# Prepare results in .csv files for markdown
print("Preparing results for markdown")
get_inputs()

# Print message that pipeline has finished
print("Pipeline has finished. Time to write this up.")