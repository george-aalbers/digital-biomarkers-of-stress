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

from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle

def predict(model, X_test, y_test):
    
    # Make predictions
    y_pred = model.predict(X_test)
        
    return y_test, y_pred

def transform_prediction_objects(X_test, y_test, y_pred):
    # Convert to dataframe
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    X_test = pd.DataFrame(X_test)
    
    # Change indices
    y_pred.index = X_test.index
    y_test.index = X_test.index
    
    # Reset index
    y_pred.reset_index(inplace=True)
    y_test.reset_index(inplace=True)
    
    # Rename columns
    y_pred.columns = ["id","y_pred"]
    y_test.columns = ["id","y_test"]
    
    # Merge prediction objects
    df = pd.merge(y_pred, y_test, on = "id", how = "inner")
    
    # Return dataframe with ids, predictions, targets
    return df

def evaluate(y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    return MAE

def evaluate_single_subject(df):
    return evaluate(df.y_test, df.y_pred)

def evaluate_single_model(model, X_test, y_test, study_parameters, loop_id):
    
    # Predict
    y_test, y_pred = predict(model, X_test, y_test)
    
    # Transform prediction objects
    df = transform_prediction_objects(X_test, y_test, y_pred)
    
    # Evaluate accuracy overall
    metrics_overall = evaluate(df.y_test, df.y_pred)
    
    # Evaluate accuracy for each subject
    metrics_per_subject = df.groupby("id").apply(evaluate_single_subject)
    
    # Write results to file
    metrics_per_subject.to_csv(study_parameters["results_output_path"] + "MAE_per_subject.csv", mode = "a", header = None)
    
def mae_nomothetic_models():
    study_parameters = pd.read_json("study_parameters.json")
    model_names = ["lasso","svr","rf"]
    for n, experiment in study_parameters.iloc[:3,:].iterrows():
        for fold in range(1,6,1):
            X_test = pd.read_csv(experiment["data_output_path"] + "X_test_" + str(fold) + ".csv", index_col = 0)
            y_test = pd.read_csv(experiment["data_output_path"] + "y_test_" + str(fold) + ".csv", index_col = 0)
            model = pickle.load(open(experiment["model_output_path"] + model_names[n] + "_" + str(fold) + ".pkl", 'rb'))
            evaluate_single_model(model, X_test, y_test, experiment, fold)     

def mae_idiographic_models():
    participant_list = pd.read_csv("data.csv").id.unique()
    study_parameters = pd.read_json("study_parameters.json")
    model_names = ["lasso","svr","rf"]
    for n, experiment in study_parameters.iloc[3:,:].iterrows():
        for participant_id in participant_list:
            X_test = pd.read_csv(experiment["data_output_path"] + "X_test_" + str(participant_id) + ".csv", index_col = 0)
            y_test = pd.read_csv(experiment["data_output_path"] + "y_test_" + str(participant_id) + ".csv", index_col = 0)
            model = pickle.load(open(experiment["model_output_path"] + model_names[n-3] + "_" + str(participant_id) + ".pkl", 'rb'))
            evaluate_single_model(model, X_test, y_test, experiment, participant_id)               