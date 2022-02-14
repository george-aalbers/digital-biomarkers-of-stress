# Functions for preprocessing of the data

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold   
import time
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit, GroupKFold
from build_model import build_model
from train_model import train_model
from write_to_file import write_to_file

def feature_extraction(esm_data, log_data, study_parameters):    
    '''
    Description
    ---
    This function does feature extraction from smartphone application usage data.

    Input
    ---
    param study_specification: a vector with the feature extraction specification of one experiment.

    Output
    ---
    The output of this script is a set of features for one specific experiment.
    ---
    '''
    
    # Import previously defined functions
    from feature_extraction_tools import asof_merge, recalculate_duration, audit_asof_merge, calculate_frequency, rename_feature_columns, merge_application_usage_features, select_categories, categorize_applications, define_time_windows, extract_smartphone_application_usage_features
    from sleep_features import sleep_features
    from temporal_features import temporal_features
    import pandas as pd
    import numpy as np
    import os

    # Extract smartphone application usage features
    df = extract_smartphone_application_usage_features(esm_data, log_data, study_parameters)

    # Extract and add sleep features
    df = sleep_features(df, study_parameters)

    # Extract and add temporal features
    df = temporal_features(df)
    
    # Write data to file
    df.to_csv("data.csv")

def aggregate_targets(study_parameters):
    
    # Read data
    df = pd.read_csv("data.csv", index_col = 0)
    
    # Aggregate targets
    df[study_parameters["targets"]] = df[study_parameters["non_aggregated_targets"]].mean(axis = 1)
    
    # Write to file
    df.to_csv("data.csv")

def drop_participants(study_parameters):
    
    # Read data
    data = pd.read_csv("data.csv", index_col = 0)
    
    # Drop missingness
    data.dropna(inplace = True)
    
    # Calculate number of observations per participant
    n_obs = data.id.value_counts()
    
    # Get the ids of participants who have less than 5 observations
    pp_ids = n_obs[n_obs > 5].index.values.tolist()
    
    # Drop the other participants
    data_dropped_pps = data[data.id.isin(pp_ids)]
    
    # Write the resulting data to file
    data_dropped_pps.to_csv("data.csv")
    
def transform_to_dataframe(df):
    return pd.DataFrame(df)
    
def select_features_targets_ids(study_parameters):
    
    # Read preprocessed data
    data = pd.read_csv("data.csv", index_col = 0, low_memory = False)
    
    # Split into X and y
    ids = data[study_parameters["id_variable"]]
    X = data[study_parameters["features"]]
    y = data[study_parameters["targets"]]
    
    # Set X and y index to ids
    y.index = ids
    X.index = ids
    
    # Write to file
    ids.to_csv("ids.csv")
    X.to_csv("X.csv")
    y.to_csv("y.csv")
    
def center_all_data_single_subject(df, study_parameters):
    
    # Center all data for one participant
    df.iloc[:,1:] = df.iloc[:,1:] - df.iloc[:,1:].mean()
    
    return df
    
def within_person_center_nomothetic(train, test, study_parameters):
    
    # Transform data to dataframe
    train = transform_to_dataframe(train).reset_index()
    test = transform_to_dataframe(test).reset_index()
    
    # Center all data for multiple participants
    train = train.groupby("id").apply(center_all_data_single_subject, study_parameters)
    test = test.groupby("id").apply(center_all_data_single_subject, study_parameters)
    
    # Set index to id variable
    train.set_index("id", inplace=True)
    test.set_index("id", inplace=True)
    
    return train, test
    
def within_person_center_idiographic(train, test, study_parameters):
    
    # Center one person's data based on mean in train data
    
    # Get mean in train data
    mean = train.mean()
    
    # Subtract from train data
    train = train - mean
    
    # And subtract from test data
    test = test - mean
    
    return train, test
        
def center(train, test, study_parameters):
    
    # Center data for each person, method depends on experiment type
    if study_parameters["experiment_type"] == "idiographic":
        return within_person_center_idiographic(train, test, study_parameters)
    elif study_parameters["experiment_type"] == "nomothetic":
        return within_person_center_nomothetic(train, test, study_parameters)

def flatten_targets(df):
    return np.ravel(df)

def drop_superfluous_columns(study_parameters):
    
    # Read data
    df = pd.read_csv("data.csv", index_col = 0)
    
    # Get features, targets, and id variable names
    variables = study_parameters["features"]
    variables.append(study_parameters["targets"])
    variables.append(study_parameters["id_variable"])
    
    # Reverse variable names
    variables.reverse()
    
    # Select those variables
    df = df[variables]
    
    # Write to file
    df.to_csv("data.csv") 
 
def drop_invariant_features(df): 
    selector = VarianceThreshold()
    features = selector.fit_transform(df.iloc[:,1:])
    features = pd.DataFrame(features)
    features.columns = selector.get_feature_names_out()
    ids = df.iloc[:,0]
    ids.reset_index(inplace=True,drop=True)
    df = pd.concat([ids, pd.DataFrame(features)], axis = 1)
    return df

def idiographic_experiment(study_parameters):
    
    X = pd.read_csv("X.csv", index_col = 0)
    y = pd.read_csv("y.csv", index_col = 0)
    ids = pd.read_csv("ids.csv", index_col = 0)
    
    # Get total sample size
    sample_size = ids.nunique()

    # Set index to ids
    X.index = y.index
    
    # Ravel ids
    ids = np.ravel(ids)
    
    # Leave one participant out
    logo = LeaveOneGroupOut()

    # Select data for one participant
    for train_index, test_index in logo.split(X, y, ids):
        X_single_subject = X.iloc[test_index, :]
        y_single_subject = y.iloc[test_index]
        
        # Get participant ID
        loop_id = X_single_subject.index.unique()[0]
           
        if study_parameters["time_series_k_splits"] == 1:

            # Determine size of test set
            test_size = np.round(X_single_subject.shape[0] * study_parameters["time_series_test_size"], decimals = 0).astype(int)

            # Split the participant's data into train and test
            X_train, X_test = X_single_subject.iloc[:-test_size, :], X_single_subject.iloc[-test_size:, :]
            y_train, y_test = y_single_subject.iloc[:-test_size], y_single_subject.iloc[-test_size:]

            # Get id column
            ids_train = X_train.index
            ids_test = X_test.index
            
            # Drop invariant features
            X_train = drop_invariant_features(X_train)
            X_test = X_test[X_train.columns]

            # Get feature names
            names = X_train.columns
                        
            # Scale features
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test) 
            
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
            X_train.columns = names
            X_test.columns = names
            
            X_train.index = ids_train
            X_test.index = ids_test
            
            print("Centering the data")
            print("===")
            print("===")
            print("===")          

            # Center their data
            y_train, y_test = center(y_train, y_test, study_parameters)

            print("Writing data to file")
            print("===")
            print("===")
            print("===") 

            # Write to file
            write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters)
            
        else:
            
            print("Not implemented for this study.")

def nomothetic_experiment(study_parameters):
    
    X = pd.read_csv("X.csv", index_col = 0)
    y = pd.read_csv("y.csv", index_col = 0)
    ids = pd.read_csv("ids.csv", index_col = 0)
    
    # For each iteration in the outer loop, we leave out nomothetic_test_size individuals.
    group_kfold = GroupKFold(n_splits = study_parameters["outer_loop_cv_k_folds"])
    
    # Set loop id to zero
    loop_id = 0
    
    # We split the data, leaving out nomothetic_test_size individuals each time.
    for train_index, test_index in group_kfold.split(X, y, ids):
        
        # Split into train and test data
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Get ids 
        ids_train = X_train.index
        ids_test = X_test.index
        
        # Update the loop id
        loop_id += 1

        # Drop invariant features
        X_train = drop_invariant_features(X_train)
        X_test = X_test[X_train.columns]  

        # Get feature names
        names = X_train.columns     
        
        # Scale features
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)        

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
            
        X_train.columns = names
        X_test.columns = names
        
        X_train.index = ids_train
        X_test.index = ids_test
        
        print("Centering the data")
        print("===")
        print("===")
        print("===")          
        
        # Center their data
        y_train, y_test = center(y_train, y_test, study_parameters)   
        
        # Flatten the targets
        y_train = flatten_targets(y_train)
        y_test = flatten_targets(y_test)

        print("Writing data to file")
        print("===")
        print("===")
        print("===") 
        
        # Write all data to file
        write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters)
        
def split(study_parameters):
    if study_parameters["experiment_type"] == "idiographic":
        idiographic_experiment(study_parameters)
    elif study_parameters["experiment_type"] == "nomothetic":
        nomothetic_experiment(study_parameters)
    else:
        
        print("What are you doing?")