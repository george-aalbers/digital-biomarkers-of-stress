import pandas as pd

def write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters):
    
    # Convert all numpy objects to dataframe
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    # Set index to participant IDs
    y_train.index = X_train.index
    y_test.index = X_test.index
    
    # Write to file
    X_train.to_csv(study_parameters["data_output_path"] + "X_train_" + str(loop_id) + ".csv")
    X_test.to_csv(study_parameters["data_output_path"] + "X_test_" + str(loop_id) + ".csv")
    y_train.to_csv(study_parameters["data_output_path"] + "y_train_" + str(loop_id) + ".csv")
    y_test.to_csv(study_parameters["data_output_path"] + "y_test_" + str(loop_id) + ".csv")