# Import packages
import pandas as pd
import numpy as np
import os
from compare_models_to_baseline import comparison
from spearman_rho import get_number_of_negative_correlations, get_number_of_large_correlations
from explain_model import figure_1, figure_2, figure_3

study_parameters = pd.read_json("study_parameters.json")

# Write to file
directory = os.getcwd()
folder = directory + '/markdown/'

def table_1(feature_categories, features):
    
    table = pd.DataFrame({"Feature categories":feature_categories, "Features":features})
    table.to_csv(folder + "table-1.csv")

def table_2(categories, examples):
    
    table = pd.DataFrame({"Category":categories, "Examples":examples})
    table.to_csv(folder + "table-2.csv")
    
def table_3():
    
    study_parameters = pd.read_json("study_parameters.json")
    
    result_multiple_studies = pd.DataFrame()
    
    for index, study in study_parameters.iterrows():        
        
        mae = pd.read_csv(study["results_output_path"] + "MAE_per_subject.csv", header = None)
        rho = pd.read_csv(study["results_output_path"] + "rho_per_subject.csv", index_col = 0, header = None)
        
        mae.columns = ["id", "mae"]

        rho.reset_index(inplace=True, drop=True)
        
        rho.columns = ["id", "rho", "p"]

        result_single_study = pd.DataFrame({"model":study["experiment_type"] + "_" + study["model_type"],
                                            "person_specific_MAE_median":mae.mae.median(), 
                                            "person_specific_MAE_min":mae.mae.min(),
                                            "person_specific_MAE_max":mae.mae.max(),
                                            "person_specific_rho_median":rho.rho.median(),
                                            "person_specific_rho_min":rho.rho.min(),
                                            "person_specific_rho_max":rho.rho.max(),
                                            "% significant":float(sum(rho.p < 0.05))/float(rho.shape[0])}, index = [index])

        result_multiple_studies = pd.concat([result_multiple_studies, result_single_study], axis = 0)    
        
    comparisons = comparison(study_parameters)
    
    table = pd.merge(result_multiple_studies, comparisons, on = "model", how = "outer")
    
    df = pd.read_csv(study_parameters["baseline_path"][0], index_col = 0)
    median = df.groupby("experiment_type").median()
    mini = df.groupby("experiment_type").min()
    maxi = df.groupby("experiment_type").max()

    baseline_idiographic_median = median.iloc[0,0]
    baseline_nomothetic_median = median.iloc[1,0]

    baseline_idiographic_minimum = mini.iloc[0,1]
    baseline_nomothetic_minimum = mini.iloc[1,1]

    baseline_idiographic_maximum = maxi.iloc[0,1]
    baseline_nomothetic_maximum = maxi.iloc[1,1]
    
    table.person_specific_MAE_median[table.model == "idiographic_baseline"] = baseline_idiographic_median
    table.person_specific_MAE_median[table.model == "nomothetic_baseline"] = baseline_idiographic_median
    table.person_specific_MAE_min[table.model == "idiographic_baseline"] = baseline_idiographic_minimum
    table.person_specific_MAE_min[table.model == "nomothetic_baseline"] = baseline_idiographic_minimum
    table.person_specific_MAE_max[table.model == "idiographic_baseline"] = baseline_idiographic_maximum
    table.person_specific_MAE_max[table.model == "nomothetic_baseline"] = baseline_idiographic_maximum
        
    table.to_csv(folder + "table-3.csv")

def average_feature_importance_nomothetic_models(mean_absolute_shapley_values_path):
    df = pd.read_csv(mean_absolute_shapley_values_path)
    df.rename({"Unnamed: 0":"model"}, axis = 1, inplace = True)
    return df.groupby("model").mean()

def get_ranking(df):
    df_transpose = df.transpose()
    df_transpose.columns = ["feature_importance"]
    return df_transpose.iloc[1:].sort_values(by=["feature_importance"], ascending=False)

def table_4():
    path = "feature_importance_nomothetic_models.csv"
    average_feature_importance_per_model = average_feature_importance_nomothetic_models(path).reset_index()
    average_feature_importance_per_model.groupby("model").apply(get_ranking).to_csv(folder + "table-4.csv")
    
def mae_histogram(study_parameters):

    result_multiple_studies = pd.DataFrame()
    
    for index, study in study_parameters.iterrows():
       
        person_specific_results = pd.read_csv(study["results_output_path"] + "metrics_per_subject.csv", header = None)
        person_specific_results = person_specific_results.assign(experiment_type = study["experiment_type"])
        person_specific_results = person_specific_results.assign(model_type = study["model_type"])
        person_specific_results.columns = ["id","mae", "experiment_type", "model_type"]
        result_multiple_studies = pd.concat([result_multiple_studies, person_specific_results], axis = 0)
            
    result_multiple_studies.to_csv(folder + "figure-1.csv")

mdd_symptoms = ['Sadness',
                'Pessimism',
                'Past failure',
                'Loss of pleasure',
                'Guilty feelings',
                'Punishment feelings',
                'Self-dislike',
                'Self-criticalness',
                'Suicidal thoughts or wishes',
                'Crying',
                'Agitation',
                'Loss of interest',
                'Indecisiveness',
                'Worthlessness',
                'Loss of energy',
                'Changes in sleeping pattern',
                'Irritability',
                'Changes in appetite',
                'Concentration difficulty',
                'Tiredness or fatigue',
                'Loss of interest in sex']

def feature_importance_per_model(df):
    
    return df.mean().sort_values(ascending=False)

def table_5():
    
    # Read mean absolute Shapley values per model
    shapley_values_idiographic_models = pd.read_csv("feature_importance_idiographic_models.csv", index_col = 0)
    
    # Select models with nonzero feature importance across people
    nonzero_feature_importance = shapley_values_idiographic_models.columns[shapley_values_idiographic_models.sum() != 0]
    shapley_values_idiographic_models_nonzero_feature_importance = shapley_values_idiographic_models[nonzero_feature_importance]
    
    # Groupby model and calculate average feature importance
    feature_importance_ranking = shapley_values_idiographic_models_nonzero_feature_importance.groupby("model").apply(feature_importance_per_model)
    feature_importance_ranking = feature_importance_ranking.reset_index()
    feature_importance_ranking.columns = ['model', 'feature', 'importance']
    
    # Write to file as table 5
    feature_importance_ranking.to_csv(folder + "table-5.csv")

def get_first_depression_survey_single_participant(df):
    return df.iloc[0,:]

def in_text():

    # Read data
    df = pd.read_csv("data.csv", low_memory=False)
    mdna = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", usecols = ["id", "startTime"], low_memory=False)
    baseline = pd.read_csv("/home/haalbers/dissertation/baseline-longitudinal-clean.csv", index_col = 0, low_memory=False)
    study_parameters = pd.read_json("study_parameters.json")
    
    # Add date so we can count days person has been in the study
    mdna["date"] = pd.to_datetime(mdna.startTime).dt.date 

    # Modify depression symptom sleep problems back to BDI-II scoring
    baseline['Changes in sleeping pattern'].replace({3:2, 4:3, 5:3, 6:4, 7:4}, inplace = True) 
    
    # Select included participants
    baseline = baseline[baseline.id.isin(df.id.unique().tolist())]
    
    # Get general descriptives
    n_participants = df.id.nunique()
    n_observations = df.shape[0]
    median_compliance = df.id.value_counts().median()
    std_compliance = df.id.value_counts().std()
    hours_of_logging = ( 24 * mdna.groupby('id').date.nunique().median() )    
    baseline = baseline.groupby('id').mean().reset_index()
    percentage_female = baseline.sex.value_counts().max()/n_participants * 100
    median_age = baseline.age.median()
    std_age = baseline.age.std()

    # Results section
    
    # Number of negative correlations
    n_negative_correlations_idiographic_models = get_number_of_negative_correlations("idiographic", study_parameters)
    n_negative_correlations_nomothetic_models = get_number_of_negative_correlations("nomothetic", study_parameters)

    # Number of moderate correlations
    idiographic_models_n_large_correlations = get_number_of_large_correlations("idiographic", study_parameters, 0.3)
    nomothetic_models_n_large_correlations = get_number_of_large_correlations("nomothetic", study_parameters, 0.3)
    n_moderate_correlations_idiographic_models = idiographic_models_n_large_correlations
    n_moderate_correlations_nomothetic_models = nomothetic_models_n_large_correlations

    # Number of large correlations
    idiographic_models_n_large_correlations = get_number_of_large_correlations("idiographic", study_parameters, 0.5)
    nomothetic_models_n_large_correlations = get_number_of_large_correlations("nomothetic", study_parameters, 0.5)
    n_large_correlations_idiographic_models = idiographic_models_n_large_correlations
    n_large_correlations_nomothetic_models  = nomothetic_models_n_large_correlations
    
    # Generate table with in-text values
    in_text_values = pd.DataFrame({"variable_name" : ["n_participants", "n_observations", "median_compliance", "std_compliance", "hours_of_logging", "median_age", "std_age",
                                                      "percentage_female", 
                                                      "n_negative_correlations_idiographic_models","n_negative_correlations_nomothetic_models",
                                                      "n_moderate_correlations_idiographic_models", "n_moderate_correlations_nomothetic_models", "n_large_correlations_idiographic_models",
                                                      "n_large_correlations_nomothetic_models"], 
                                   "value" : [n_participants, n_observations, median_compliance, std_compliance, hours_of_logging, median_age, std_age, percentage_female,
                                              n_negative_correlations_idiographic_models, n_negative_correlations_nomothetic_models,
                                              n_moderate_correlations_nomothetic_models, n_moderate_correlations_nomothetic_models, n_large_correlations_idiographic_models,
                                              n_large_correlations_nomothetic_models]})
    
    # Write to file
    in_text_values.to_csv(folder + "in_text_values.csv")

def get_inputs():
    
    # Table 1
    table_1(["Time","Time","Time","Time", "Sleep", "Sleep", "Smartphone application use", "Smartphone application use"],
            ["Hour of day (0 to 23, starting at midnight)", "Day of week (0 = weekday, 1 = weekend)", "Day of month (0 to 31)", "Month (pre-COVID = 0, during COVID = 1)",
             "Sleep onset (hours)", "Sleep duration (hours)",
             "Duration (seconds) spent on smartphone application category X in the past 60 minutes", "Frequency (count) of opening smartphone application category X in the past 60 minutes"]) 

    # Table 2
    table_2(["Browser",
             "Calling",
             "Camera",
             "Dating",
             "E-mail",
             "Exercise",
             "Food & Drink",
             "Gallery",
             "Game",
             "Messenger",
             "Music & Audio",
             "Productivity",
             "Shared transportation",
             "Social networks",
             "Video",
             "Weather",
             "Work"],
             ["Chrome, Opera",
             "Default dial applications",
             "Default camera applications",
             "Tinder, Grindr",
             "Gmail, Outlook",
             "RunKeeper",
             "UberEATS",
             "Default gallery applications",
             "CandyCrush",
             "Whatsapp",
             "Spotify",
             "Microsoft Word",
             "9292OV ",
             "Facebook, Instagram",
             "Youtube, Netflix",
             "Default weather applications",
             "StudentJob, EmployeeApp"])

    # Table 3
    table_3()
    
    # Table 4
    table_4()
    
    # Table 5
    table_5()
    
    # Figure 1
    figure_1()
    
    # Figure 2
    figure_2()
    
    # Figure 3 
    figure_3()
    
    # In-text values
    in_text()