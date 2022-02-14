'''
Description
---
This function evaluates a model on out-of-fold data.

Input
---
param model: trained model
param X_test: features of the test data
param y_test: targets of the test data
param study_parameters: vector specifying how the evaluation is conducted
param loop_id: value representing the ID of the loop (eg participant ID or split number)

Output
---
The script appends a line to a .csv file that summarizes how well the model performs.
---
'''

import pandas as pd
import os 
import pickle
from scipy.stats import spearmanr

def predict(model, X_test, y_test):
    
    # Make predictions
    y_pred = model.predict(X_test)
        
    return y_test, y_pred

def evaluate(y_test, y_pred, pp_id):
    
    #calculate Spearman Rank correlation and corresponding p-value
    rho, p = spearmanr(y_test, y_pred)   
    
    return pd.DataFrame({"rho":rho, "p":p}, index = [pp_id])

def merge(y_test, y_pred, X_test):
    
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    
    y_test.index = X_test.index
    y_pred.index = X_test.index
    
    df = pd.concat([y_test, y_pred], axis = 1)
    
    df.reset_index(inplace=True)
    
    df.columns = ["id", "y_test", "y_pred"] 
    
    return df

def evaluate_single_subject(df):
    return evaluate(df.y_test, df.y_pred, df.id.unique())

def evaluate_single_model(model, X_test, y_test, study_parameters, loop_id):
    
    # Predict
    y_test, y_pred = predict(model, X_test, y_test)
    
    # Merge
    df = merge(y_test, y_pred, X_test)
    
    # Evaluate accuracy overall
    metrics_overall = evaluate(df.y_test, df.y_pred, df.id.unique())
    
    # Evaluate accuracy for each subject
    metrics_per_subject = df.groupby("id").apply(evaluate_single_subject)
    
    # Reset index
    metrics_per_subject.reset_index(inplace=True)
    
    # Drop level_1
    metrics_per_subject.drop({"level_1"}, axis = 1, inplace = True)
    
    # Write results to file
    metrics_per_subject.to_csv(study_parameters["results_output_path"] + "rho_per_subject.csv", mode = "a", header = None)
    
def rho_nomothetic_models():
    study_parameters = pd.read_json("study_parameters.json")
    model_names = ["lasso","svr","rf"]
    for n, experiment in study_parameters.iloc[:3,:].iterrows():
        for fold in range(1,6,1):
            X_test = pd.read_csv(experiment["data_output_path"] + "X_test_" + str(fold) + ".csv", index_col = 0)
            y_test = pd.read_csv(experiment["data_output_path"] + "y_test_" + str(fold) + ".csv", index_col = 0)
            model = pickle.load(open(experiment["model_output_path"] + model_names[n] + "_" + str(fold) + ".pkl", 'rb'))
            evaluate_single_model(model, X_test, y_test, experiment, fold)     

def rho_idiographic_models():
    participant_list = pd.read_csv("data.csv").id.unique()
    study_parameters = pd.read_json("study_parameters.json")
    model_names = ["lasso","svr","rf"]
    for n, experiment in study_parameters.iloc[3:,:].iterrows():
        for participant_id in participant_list:
            X_test = pd.read_csv(experiment["data_output_path"] + "X_test_" + str(participant_id) + ".csv", index_col = 0)
            y_test = pd.read_csv(experiment["data_output_path"] + "y_test_" + str(participant_id) + ".csv", index_col = 0)
            model = pickle.load(open(experiment["model_output_path"] + model_names[n-3] + "_" + str(participant_id) + ".pkl", 'rb'))
            evaluate_single_model(model, X_test, y_test, experiment, participant_id)               

def get_number_of_negative_correlations(analysis_type, study_parameters):
    
    if analysis_type == "idiographic":

        lasso = pd.read_csv(study_parameters.loc[3,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        svr = pd.read_csv(study_parameters.loc[4,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        rf = pd.read_csv(study_parameters.loc[5,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        
        df = [lasso, svr, rf]
        df = pd.concat(df, axis = 0)
        
        df.columns = ["id","rho","p"]
        
        df = df[df.p < 0.05]
        
    else:
        
        lasso = pd.read_csv(study_parameters.loc[0,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        svr = pd.read_csv(study_parameters.loc[1,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        rf = pd.read_csv(study_parameters.loc[2,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        
        df = [lasso, svr, rf]
        df = pd.concat(df, axis = 0)
        
        df.columns = ["id","rho","p"]
        
        df = df[df.p < 0.05]
    
    return df[df.rho < 0].id.nunique()

def get_number_of_large_correlations(analysis_type, study_parameters, effect_size):
    
    if analysis_type == "idiographic":

        lasso = pd.read_csv(study_parameters.loc[3,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        svr = pd.read_csv(study_parameters.loc[4,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        rf = pd.read_csv(study_parameters.loc[5,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        
        df = [lasso, svr, rf]
        df = pd.concat(df, axis = 0)

        df.columns = ["id","rho","p"]
        
        df = df[df.p < 0.05]
        
    else:
        
        lasso = pd.read_csv(study_parameters.loc[0,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        svr = pd.read_csv(study_parameters.loc[1,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        rf = pd.read_csv(study_parameters.loc[2,"results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        
        df = [lasso, svr, rf]
        df = pd.concat(df, axis = 0)

        df.columns = ["id","rho","p"]
        
        df = df[df.p < 0.05]
    
    return df[df.rho >= effect_size].id.nunique()