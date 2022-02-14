# Import pandas 
import pandas as pd

# Import modules
import pandas as pd
import numpy as np

# Functions for estimating sleep in one person
def binarize(df):
    
    # Set startTime as the index
    df.loc[:,"indexTime"] = pd.to_datetime(df.startTime)
    df.set_index("indexTime", inplace=True)

    # Make startTime and endTime a datetime object
    df.loc[:,"endTime"] = pd.to_datetime(df.endTime).tz_localize(None)
    df.loc[:,"startTime"] = pd.to_datetime(df.startTime).tz_localize(None)

    # Resample the dataframe to minutes
    df = df.groupby('id').resample("min").agg({"startTime":np.min, "endTime":np.max})

    # Forward fill startTime and endTime
    df.startTime = df.startTime.fillna(method="ffill")
    df.endTime = df.endTime.fillna(method="ffill")

    # Reset the index
    df.reset_index(inplace=True)

    # Replace instances where indexTime exceeds startTime by indexTime
    df.startTime[df.indexTime > df.startTime] = df.indexTime

    # Set the index to indexTime again
    df.set_index(["id","indexTime"], inplace = True)

    # Create an all-zero column called binary bins, representing whether a person used their phone or not
    df.loc[:,"binary_bins"] = 0

    # Replace all zeros with a one if startTime is smaller than endTime (meaning a person used their phone)
    df.loc[:,"binary_bins"][df.startTime < df.endTime] = 1
    
    return df

def get_least_activity(df, timescale = 15, n_windows = 24):
    
    # Transform timescale and time window to string
    timescale_str = str(timescale) + "Min"
    timewindow_str = str(timescale*n_windows) + "Min"
    
    # Resample dataframe to 15 minute timescale
    least_activity = df.reset_index().set_index("indexTime").resample(timescale_str).sum()
    least_activity.reset_index(inplace=True)
    least_activity["time"] = least_activity.indexTime.dt.time
    least_activity = pd.concat([least_activity.groupby("time").mean(), 
                                least_activity.groupby("time").mean()]).rolling(n_windows).mean().dropna().sort_values(by=["binary_bins"])
    
    # Determine stop and start of least activity time window and concatenate
    stop = pd.to_datetime(least_activity.reset_index().time.astype(str)[0])
    start = stop - pd.Timedelta(timewindow_str)
    stop_start = pd.to_datetime(pd.concat([pd.Series(stop), pd.Series(start)])).dt.time
    
    return stop_start

def calculate_moving_sum(df, window_size = 120, edge_thickness = 60):
    
    # Create moving sum of binary states (120 minute time window with epoch at center)
    log_data_moving_sum = df.rolling(window_size, center = True).sum()

    # Drop generated missingness
    log_data_moving_sum.dropna(inplace=True)

    # Reset the index again
    log_data_moving_sum.reset_index(inplace=True)

    # Create string variable for edge thickness 
    edge_thickness_str = edge_thickness + "Min"
    
    # Add edges of the time window
    log_data_moving_sum["startTime"] = log_data_moving_sum["indexTime"] - pd.Timedelta(edge_thickness_str)
    log_data_moving_sum["endTime"] = log_data_moving_sum["indexTime"] + pd.Timedelta(edge_thickness_str)

    return log_data_moving_sum

def select_potential_sleep_epochs(df, log_data_moving_sum, threshold = 2):

    # Select bins with subthreshold smartphone usage
    potential_sleep_epochs = log_data_moving_sum[log_data_moving_sum["binary_bins"] < threshold].dropna()
    
    # Create a date variable so we can group according to date
    potential_sleep_epochs.loc[:,"date"] = pd.to_datetime(potential_sleep_epochs["indexTime"]).dt.date.astype(str)

    # Select hour variable so we can deselect hours outside rest range
    potential_sleep_epochs.loc[:,"hour"] = pd.to_datetime(potential_sleep_epochs["indexTime"]).dt.hour
    
    # Select startTime/endTime combinations that overlap with stop/start
    # If either the startTime or the endTime is between the regular sleep time, then we assume the person is sleeping
    starttime = potential_sleep_epochs["startTime"].dt.time.between(least_activity.iloc[1], least_activity.iloc[0])
    endtime = potential_sleep_epochs["endTime"].dt.time.between(least_activity.iloc[1], least_activity.iloc[0])

    # We select rows where we believe people are sleeping
    potential_sleep_epochs = potential_sleep_epochs[(starttime) | (endtime)]

    return potential_sleep_epochs
 
def sleep_detection(df):
    return df.groupby("date").startTime.min(), df.groupby("date").endTime.max()

def merge_sleep_onset_offset(sleep_onset, sleep_offset):
    return pd.merge(sleep_onset, sleep_offset, on = "date", how = "outer")
    
def calculate_sleep_duration(df):
    df["duration"] = df["endTime"] - df["startTime"]
    return df
    
def sleep_single_subject(data, timescale = 15, n_windows = 24, window_size = 120, edge_thickness = 60, threshold = 2):
    df = data.copy(deep=True)
    
    # Binarize the time-series
    df = binarize(df)

    # Get person-specific least activity
    least_activity = get_least_activity(df, timescale, n_windows)

    # Calculate moving sum
    moving_sum = calculate_moving_sum(df, window_size, edge_thickness)

    # Select potential sleep epochs
    potential_sleep_epochs = select_potential_sleep_windows(df, moving_sum, threshold)

    # We then get per date the minimum startTime and maximal endTime
    sleep_onset, sleep_offset = sleep_detection(potential_sleep_epochs)

    # Merge sleep onset and offset
    sleep = merge_sleep_onset_offset(sleep_onset, sleep_offset)

    # Calculate sleep duration
    sleep = calculate_sleep_duration(sleep)

    return sleep

# Define function for calculating all sleep features
def calculate_sleep_features(study_parameters):

    # Read modules
    import pandas as pd

    # Read data
    data = pd.read_csv(study_parameters["sleep_features_path"], index_col = 0)
    data.columns = ["id","date","sleep_onset","wake_time","sleep_duration"]

    # Convert datetime objects to floats
    data.sleep_onset = pd.to_datetime(data.sleep_onset).dt.hour + pd.to_datetime(data.sleep_onset).dt.minute/60
    data.wake_time = pd.to_datetime(data.wake_time).dt.hour + pd.to_datetime(data.wake_time).dt.minute/60
    data.sleep_duration = pd.to_datetime(data.sleep_duration.str[8:]).dt.hour + pd.to_datetime(data.sleep_duration.str[8:]).dt.minute/60
    data.sleep_onset = data.wake_time - data.sleep_duration

    # Forward shift post-midnight sleep 24 hours as well as all wake times
    data.sleep_onset = data.sleep_onset[data.sleep_onset < 10] + 24
    data.wake_time = data.wake_time + 24
        
    # Drop missingness
    data.dropna(inplace=True)

    return data

# Define function for adding sleep features
def add_sleep_features(main_df, sleep_df):
    main_df["Response Time_ESM_day"] = pd.to_datetime(main_df["Response Time_ESM_day"].str[:19])
    main_df["date"] = pd.to_datetime(main_df["Response Time_ESM_day"].dt.date)
    sleep_df["date"] = pd.to_datetime(sleep_df["date"])
    
    df = pd.merge(main_df, sleep_df, on = ["id","date"], how = "inner")
    
    return df

# Define function for doing all the above
def sleep_features(main_df, study_parameters):
    
    sleep_df = calculate_sleep_features(study_parameters)
    df = add_sleep_features(main_df, sleep_df)
    df.rename({"sleep_onset":"Sleep onset", "sleep_duration":"Sleep duration"}, axis = 1, inplace = True)
    
    return df