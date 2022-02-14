# Define function for getting temporal features
def temporal_features(df):
    
    # Import pandas
    import pandas as pd
    
    # Extract temporal features
    df["Hour of day"] = df["Response Time_ESM_day"].dt.hour + df["Response Time_ESM_day"].dt.minute/60
    df["dayofweek"] = df["Response Time_ESM_day"].dt.dayofweek
    df["Day of week"] = df["dayofweek"].replace({0:0,1:0,2:0,3:0,4:0,5:1,6:1})    
    df["Day of month"] = df["Response Time_ESM_day"].dt.day
    df["month"] = df["Response Time_ESM_day"].dt.month
    df["COVID-19"] = df["month"].replace({0:0,1:0,2:0,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1})
    df["date"] = df["Response Time_ESM_day"].dt.date
    
    # Return df with temporal features
    return df