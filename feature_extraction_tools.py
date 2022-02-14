import numpy as np
import pandas as pd

def extract_momentary_features(esm_data, log_data, study_parameters):
    # Create objects representing the time window we are interested in
    timedelta, timedelta_seconds = define_time_windows(study_parameters)

    # Conduct an asof merge with "forward" direction and a timedelta of X minutes. This selects all app events ending in the X-minute time window before the self-report
    duration_before_df = asof_merge(esm_data, log_data, timedelta=timedelta, direction="forward", esm_time_var = "Response Time_ESM_day")

    # Calculate frequency
    frequency_before_df = calculate_frequency(duration_before_df, esm_data, before=True, timedelta=timedelta, write_to_file=False, output_path=study_parameters["data_output_path"])

    # For most application events, this means they start and end within a X-minute time window before the self-report. Some of them, however, will start before the time window. We need 
    # to correct for this using recalculate duration.
    duration_before_df_re = recalculate_duration(duration_before_df, esm_data, before=True, timedelta=timedelta, write_to_file=False, output_path=study_parameters["data_output_path"])
    
    # Do a quick audit to check whether we have any impossible values (i.e., larger than the time window size)
    audit_asof_merge(duration_before_df_re, time_window_seconds = timedelta_seconds)

    # Categorize applications
    df1 = categorize_applications(duration_before_df_re, study_parameters)
    df2 = categorize_applications(frequency_before_df, study_parameters)

    # Rename features
    df1 = rename_feature_columns(df1, study_parameters, duration = True)
    df2 = rename_feature_columns(df2, study_parameters, duration = False)

    # Merge files
    df = merge_application_usage_features(df1, df2, study_parameters)
    
    # Fill 'missingness' in the features with zeros, as these aren't actually missing (there was simply no smartphone use before these survey responses)
    df.fillna(0, inplace = True)

    return df

def extract_smartphone_application_usage_features(esm_data, log_data, study_parameters):
    '''
    Description
    ---
    Function to extract smartphone application usage features from the log data.
    
    Parameters
    ---
    param esm_data:           dataframe with experience sampling survey responses
    param log_data:           dataframe with mobileDNA log data
    param study_parameters:   dataframe with specification of the study parameters
 
    Example
    ---
    extract_smartphone_application_usage_features(esm_data, log_data, study_parameters)
    
    '''
    ##############################################################################
    # Selecting application usage duration in the X minutes before self-reports #
    ##############################################################################
    
    df = extract_momentary_features(esm_data, log_data, study_parameters)
    
    return df

def asof_merge(esm_data, log_data, timedelta="15Min", direction="forward", esm_time_var = "Response Time_ESM_day"):
    
    '''
    Description
    ---
    Function to perform an asof merge on experience sampling survey data and mobileDNA log data.
    
    Parameters
    ---
    param esm_data:           dataframe with experience sampling survey responses
    param log_data:           dataframe with mobileDNA log data
    param timedelta:          string variable with desired time window size in minutes (e.g., "15Min" if you want to select all smartphone use in an adjacent 15 minute window)
    param direction:          string variable indicating the direction of the asof merge ("forward" for selecting usage before self-report, "backward" for selecting usage after self-report)
    param esm_time_var:       string variable indicating name of the time column in the experience sampling dataframe    
    
    Example
    ---
    asof_merge(esm_data, log_data, timedelta="15Min", direction="forward", esm_time_var = "Response Time_ESM_day")
    
    '''
    
    # Set log_time_var to startTime (for backward asof merge) or endTime (for forward asof merge)
    if direction == "backward":
        log_time_var = "startTime"
    else:
        log_time_var = "endTime"
    
    # Create "time" variable
    log_data["time"] = pd.to_datetime(log_data[log_time_var])
    esm_data["time"] = esm_data["Response Time_ESM_day"].str[:19]
    
    # Give dataframes new name
    left = log_data
    right = esm_data
    
    # Convert time to datetime
    left.time = pd.to_datetime(left.time)
    right.time = pd.to_datetime(right.time)
    
    # Transform datetime columns to same format
    left.time = left.time.dt.strftime('%Y/%m/%d %H:%M:%S')
    right.time = right.time.dt.strftime('%Y/%m/%d %H:%M:%S')

    # Convert to datetime again
    left.time = pd.to_datetime(left.time)
    right.time = pd.to_datetime(right.time) 
    
    # Sort the time values
    left = left.sort_values("time")
    right = right.sort_values("time")
    
    # Do the asof merge
    asof_merge = pd.merge_asof(left, right, on="time", by="id", suffixes=('_x', '_y'), tolerance = pd.Timedelta(timedelta), allow_exact_matches=True, direction=direction)
    
    return asof_merge
        
def recalculate_duration(asof, esm_data, before = True, timedelta = "15Min", write_to_file = True, output_path = None):

    '''
    Description
    ---
    Function to correct incorrect durations returned by the asof_merge function.
    
    Parameters
    ---
    param asof:               dataframe resulting from the asof merge
    param before:             boolean indicating whether the asof merge captured smartphone use before (True) or after (False) a self-report
    param timedelta:          string variable with time window size of asof merge
    param write_to_file:      boolean indicating whether we want the resulting data to be written to file
    param output_path:        string variable indicating path to the folder to which the features should be written   
    
    Example
    ---
    recalculate_duration(asof, before = True, timedelta = "15Min", write_to_file = True)
    
    '''
    
    if before:
    
        # Get the startTime of app usage and ReponseTime of survey
        time1 = pd.to_datetime(asof['startTime']) 
        time2 = pd.to_datetime(asof['Response Time_ESM_day'].str[:19])

        # Transform datetime vectors to same format
        time1 = time1.dt.strftime('%Y/%m/%d %H:%M:%S')
        time2 = time2.dt.strftime('%Y/%m/%d %H:%M:%S')

        # Convert to datetime again
        asof["log_start_time"] = pd.to_datetime(time1)
        asof["survey_response_time"] = pd.to_datetime(time2) 

        # Select rows with survey response time
        asof = asof[~pd.isnull(asof["survey_response_time"])]

        # Sort rows on ID and time
        asof = asof.sort_values(["id","time"])

        # Add start of window
        asof["window_start"] = asof["survey_response_time"] - pd.Timedelta(timedelta)

        # For those instances, set log start time to window start and calculate durationSeconds based on that
        asof["durationSeconds"] = (pd.to_datetime(asof["endTime"]) - pd.to_datetime(asof["startTime"])).dt.seconds

        # Loop through instances where window start is later than log start time (i.e., app started before window started)
        for i in asof[asof["window_start"] > asof["log_start_time"]]["durationSeconds"].index.tolist():

            # Replace those instances by the difference between start of time window and end of application usage
            asof.loc[i, "durationSeconds"] = (pd.to_datetime(asof.loc[i,"endTime"]) - pd.to_datetime(asof.loc[i,"window_start"])).seconds

        # Groupby id, response time, application and pivot the dataframe    
        df = asof.groupby(["id",'Response Time_ESM_day','application']).sum().reset_index().pivot(columns="application", index=["id","Response Time_ESM_day"], values = "durationSeconds").fillna(0)

        # Merge with experience sampling responses
        df = pd.merge(esm_data, df.reset_index(), on = ["id", "Response Time_ESM_day"], how = "outer")

        # Write result to file?
        if write_to_file:
            df.to_csv(output_path + "duration-before-asof-merge.csv")
            return df
        else:
            return df
    
    else:

        # Get the endTime of app usage and ReponseTime of survey
        time1 = pd.to_datetime(asof['endTime']) 
        time2 = pd.to_datetime(asof['Response Time_ESM_day'].str[:19])

        # Transform datetime vectors to same format
        time1 = time1.dt.strftime('%Y/%m/%d %H:%M:%S')
        time2 = time2.dt.strftime('%Y/%m/%d %H:%M:%S')

        # Convert to datetime again
        asof["log_end_time"] = pd.to_datetime(time1)
        asof["survey_response_time"] = pd.to_datetime(time2) 

        # Select rows with survey response time
        asof = asof[~pd.isnull(asof["survey_response_time"])]

        # Sort rows on ID and time
        asof = asof.sort_values(["id","time"])

        # Add start of window
        asof["window_end"] = asof["survey_response_time"] + pd.Timedelta(timedelta)

        # For those instances, set log start time to window start and calculate durationSeconds based on that
        asof["durationSeconds"] = (pd.to_datetime(asof["endTime"]) - pd.to_datetime(asof["startTime"])).dt.seconds

        # Loop through instances where window start is later than log start time (i.e., app started before window started)
        for i in asof[asof["window_end"] < asof["log_end_time"]]["durationSeconds"].index.tolist():

            # Replace those instances by the difference between start of time window and end of application usage (we get microseconds here as pandas goes wild otherwise)
            asof.loc[i, "durationSeconds"] = (pd.to_datetime(asof.loc[i,"window_end"]) - pd.to_datetime(asof.loc[i,"startTime"])).microseconds/1000000

        # Groupby id, response time, application and pivot the dataframe    
        df = asof.groupby(["id",'Response Time_ESM_day','application']).sum().reset_index().pivot(columns="application", index=["id","Response Time_ESM_day"], values = "durationSeconds").fillna(0)

        # Merge with experience sampling responses
        df = pd.merge(esm_data, df.reset_index(), on = ["id", "Response Time_ESM_day"], how = "outer")

        # Write result to file?
        if write_to_file:
            df.to_csv(output_path + "duration-after-asof-merge.csv")
            return df
        else:
            return df

def audit_asof_merge(df, time_window_seconds = 900):
    '''
    Description
    ---
    Function that prints maximum time on application in asof-merged dataframe and a warning message if this exceeds size of time window.
    
    Parameters
    ---
    param df:                       dataframe returned by recalculate_duration
    param time_window_seconds:      integer representing size of time window
    
    Example
    ---
    audit_asof_merge(df, time_window_seconds = 900)
    '''
    
    # Import study parameters
    study_parameters = pd.read_json("study_parameters.json")
    
    # Drop missingness in self-reports and fill missings in log data with 0 (these cells represent times people didn't use their smartphones)
    df_na_dropped = df.dropna(subset=[study_parameters["non_aggregated_targets"][0][0]]).fillna(0)
    
    # Get the max values for each row (i.e., how much time did participant spend on smartphone per time window)
    max_values = df_na_dropped.max(axis=1, numeric_only=True)
    
    # Then select the features
    max_values = max_values.iloc[27:]
    
    # And then get the maximum value of those
    maximum_duration = max_values.max()
    if maximum_duration > time_window_seconds:
        print("Maximum duration in this dataframe equals", maximum_duration, "seconds. This is larger than the time window, so something must have gone wrong.")
        print("===")
        print("===")
        print("===")
    else:
        print("Maximum duration in this dataframe equals", maximum_duration, "seconds. All good! No values that exceed time window size.")
        print("===")
        print("===")
        print("===")
        
def calculate_frequency(asof, esm_data, before = True, timedelta = "15Min", write_to_file = True, output_path = None):

    '''
    Description
    ---
    Function to calculate number of application events from dataframe returned by the asof_merge function.
    
    Parameters
    ---
    param asof:               dataframe resulting from the asof merge
    param before:             boolean indicating whether the asof merge captured smartphone use before (True) or after (False) a self-report
    param timedelta:          string variable with time window size of asof merge
    param write_to_file:      boolean indicating whether we want the resulting data to be written to file
    param output_path:        string variable indicating path to the folder to which the features should be written   

    Example
    ---
    calculate_frequency(asof, before = True, timedelta = "15Min", write_to_file = True, output_path)
    
    '''
    
    if before:
    
        # Get the startTime of app usage and ReponseTime of survey
        time1 = pd.to_datetime(asof['startTime']) 
        time2 = pd.to_datetime(asof['Response Time_ESM_day'].str[:19])

        # Transform datetime vectors to same format
        time1 = time1.dt.strftime('%Y/%m/%d %H:%M:%S')
        time2 = time2.dt.strftime('%Y/%m/%d %H:%M:%S')

        # Convert to datetime again
        asof["log_start_time"] = pd.to_datetime(time1)
        asof["survey_response_time"] = pd.to_datetime(time2) 

        # Select rows with survey response time
        asof = asof[~pd.isnull(asof["survey_response_time"])]

        # Sort rows on ID and time
        asof = asof.sort_values(["id","time"])

        # Add start of window
        asof["window_start"] = asof["survey_response_time"] - pd.Timedelta(timedelta)

        # For those instances, set log start time to window start and calculate durationSeconds based on that
        asof["durationSeconds"] = (pd.to_datetime(asof["endTime"]) - pd.to_datetime(asof["startTime"])).dt.seconds

        # Loop through instances where window start is later than log start time (i.e., app started before window started)
        for i in asof[asof["window_start"] > asof["log_start_time"]]["durationSeconds"].index.tolist():

            # Replace those instances by the difference between start of time window and end of application usage
            asof.loc[i, "durationSeconds"] = (pd.to_datetime(asof.loc[i,"endTime"]) - pd.to_datetime(asof.loc[i,"window_start"])).seconds

        # Groupby id, response time, application and pivot the dataframe    
        df = asof.groupby(["id",'Response Time_ESM_day','application']).count().reset_index().pivot(columns="application", index=["id","Response Time_ESM_day"], values = "durationSeconds").fillna(0)

        # Merge with experience sampling responses
        df = pd.merge(esm_data, df.reset_index(), on = ["id", "Response Time_ESM_day"], how = "outer")

        # Write result to file?
        if write_to_file:
            df.to_csv(output_path + "frequency-before-asof-merge.csv")
            return df
        else:
            return df
    
    else:

        # Get the endTime of app usage and ReponseTime of survey
        time1 = pd.to_datetime(asof['endTime']) 
        time2 = pd.to_datetime(asof['Response Time_ESM_day'].str[:19])

        # Transform datetime vectors to same format
        time1 = time1.dt.strftime('%Y/%m/%d %H:%M:%S')
        time2 = time2.dt.strftime('%Y/%m/%d %H:%M:%S')

        # Convert to datetime again
        asof["log_end_time"] = pd.to_datetime(time1)
        asof["survey_response_time"] = pd.to_datetime(time2) 

        # Select rows with survey response time
        asof = asof[~pd.isnull(asof["survey_response_time"])]

        # Sort rows on ID and time
        asof = asof.sort_values(["id","time"])

        # Add start of window
        asof["window_end"] = asof["survey_response_time"] + pd.Timedelta(timedelta)

        # For those instances, set log start time to window start and calculate durationSeconds based on that
        asof["durationSeconds"] = (pd.to_datetime(asof["endTime"]) - pd.to_datetime(asof["startTime"])).dt.seconds

        # Loop through instances where window start is later than log start time (i.e., app started before window started)
        for i in asof[asof["window_end"] < asof["log_end_time"]]["durationSeconds"].index.tolist():

            # Replace those instances by the difference between start of time window and end of application usage (we get microseconds here as pandas goes wild otherwise)
            asof.loc[i, "durationSeconds"] = (pd.to_datetime(asof.loc[i,"window_end"]) - pd.to_datetime(asof.loc[i,"startTime"])).microseconds/1000000

        # Groupby id, response time, application and pivot the dataframe    
        df = asof.groupby(["id",'Response Time_ESM_day','application']).count().reset_index().pivot(columns="application", index=["id","Response Time_ESM_day"], values = "durationSeconds").fillna(0)

        # Merge with experience sampling responses
        df = pd.merge(esm_data, df.reset_index(), on = ["id", "Response Time_ESM_day"], how = "outer")

        # Write result to file?
        if write_to_file:
            df.to_csv(output_path + "frequency-after-asof-merge.csv")
            return df
        else:
            return df
        
def rename_feature_columns(df, study_parameters, duration):
    if duration:
        df.rename(dict(zip(study_parameters["categories"], (pd.Series(study_parameters["categories"]) + " (duration)").tolist())), axis = 1, inplace = True)
    else:
        df.rename(dict(zip(study_parameters["categories"], (pd.Series(study_parameters["categories"]) + " (frequency)").tolist())), axis = 1, inplace = True)
    return df

def merge_application_usage_features(df1, df2, study_parameters):
    df = pd.merge(df1, df2, on = study_parameters["self_report_columns"], how = "outer")
    return df

def select_categories(df, study_parameters):
    select = df.columns[df.columns.isin(study_parameters["categories"])]
    return df[select]

def categorize_applications(df, study_parameters):
    # Load the application category dictionary
    dictionary = pd.read_csv("category_dictionary_final.csv", usecols = ["name","category"])
    
    # Rename
    df = df.rename(dict(zip(dictionary.name, dictionary.category)), axis = 1)

    # Aggregate columns with same name
    features = pd.DataFrame()
    categories = df.columns[33:].unique().tolist()
    for category in categories:
        features = pd.concat([features, df.filter(regex=category).fillna(0).sum(axis=1)], axis = 1)
    features.columns = categories

    # Select categories to be included in analysis
    features = select_categories(features, study_parameters)

    # Attach features to targets
    df = pd.concat([df.iloc[:,:33], features], axis = 1)

    # Return df
    return df

def define_time_windows(study_parameters):
    timedelta = str(study_parameters["window_size"]) + " minutes"
    print(timedelta)
    seconds = study_parameters["window_size"] * 60
    return timedelta, seconds
