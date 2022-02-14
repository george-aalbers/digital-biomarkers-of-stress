import pandas as pd
import numpy as np

study_parameters = pd.read_json("study_parameters.json")

def merge_model_results_baseline_results(study_parameters):
    
    results = pd.DataFrame()
    
    baseline = pd.read_csv(study_parameters["baseline_path"][0], index_col = 0)
    
    for index, experiment in study_parameters.iterrows():
        
        # Read model results
        model = pd.read_csv(experiment["results_output_path"] + "MAE_per_subject.csv", header = None)
        
        # Rename columns
        model.columns = ["id","mae"]
        
        # Merge baseline and model results
        both = pd.merge(baseline[baseline["experiment_type"] == experiment["experiment_type"]], model, on = ["id"], how = "outer", suffixes = ["_baseline", "_model"])
        
        # Set index to model type
        both["model_type"] = np.repeat(experiment["model_type"], both.shape[0])
        
        results = pd.concat([results, both], axis = 0)
        
    return results

def calculate_better_mae_than_baseline_single_model(df):
    
    difference = df.mae_model - df.mae_baseline
    
    percentage_better = float(sum(difference < 0))/float(len(difference)) * 100
    
    return percentage_better

def calculate_better_mae_than_baseline_multiple_models(df):
    
    # Calculate how often models are better than baseline
    better_than_baseline = df.groupby(['experiment_type', 'model_type']).apply(calculate_better_mae_than_baseline_single_model)

    # Reset index
    better_than_baseline = better_than_baseline.reset_index()
    
    # Add column signifying model type
    better_than_baseline["model"] = better_than_baseline["experiment_type"] + "_" + better_than_baseline["model_type"]
    
    # Rename columns
    better_than_baseline.columns = ["experiment_type", "model_type", "% better than baseline", "model"]
    
    # Drop columns
    better_than_baseline.drop(["experiment_type", "model_type"], axis = 1, inplace = True)

    return better_than_baseline
    
def calculate_best_model(results_model_baseline):
     
    # Get baseline performance 
    baseline = results_model_baseline[["experiment_type", "id", "mae_baseline"]].drop_duplicates()
    
    # Add column MAE model (this is the performance of the baseline model)
    baseline["mae_model"] = baseline["mae_baseline"]
    
    # Add a label to clarify that this is the baseline model
    baseline["model_type"] = np.repeat("baseline", baseline.shape[0])
    
    # Add to original dataframe
    results_model_baseline = pd.concat([results_model_baseline, baseline], axis = 0)
    
    # Create column with model name
    results_model_baseline["model"] = results_model_baseline["experiment_type"] + "_" + results_model_baseline["model_type"] 
    
    # Reset the index
    results_model_baseline.reset_index(inplace = True, drop = True)
    
    # Calculate winning model per participant
    winning_models = results_model_baseline.loc[results_model_baseline.groupby(["experiment_type", "id"]).mae_model.idxmin().values,"model"]

    # Calculate how often each model wins
    percentages = pd.DataFrame(winning_models.value_counts() / results_model_baseline.id.nunique() * 100).reset_index()
    
    # Rename columns
    percentages.columns = ["model", "% best model"]
    
    # Return sorted alphabetically
    return percentages.sort_values("model").reset_index(drop=True)

def comparison(study_parameters):

    # Merge model and baseline results
    results_model_baseline = merge_model_results_baseline_results(study_parameters)
    
    # How often are models better than baseline?
    better_than_baseline = calculate_better_mae_than_baseline_multiple_models(results_model_baseline)
    
    # How often do models outperform all other models (including baseline)?
    best_model = calculate_best_model(results_model_baseline)
    
    # Merge
    table = pd.merge(better_than_baseline, best_model, on = "model", how = "outer")
    
    # Reorder columns
    table = table[["model", "% better than baseline", "% best model"]]
    
    # Sort alphabetically
    table = table.sort_values("model")
    
    # Reset index
    table.reset_index(inplace = True, drop = True)
    
    return table