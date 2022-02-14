'''
Description
---
This function builds models for cross-validation. 

Input
---
param study_parameters: vector that specifies prediction task, model type, cross-validation type, number of jobs, number of inner CV splits

Output
---
The output of this script is an initialized model that can be cross-validated.
'''

# Import
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define function for building the model
def build_model(study_parameters):
    
    if study_parameters["prediction_task"] == "regression":
        if study_parameters["model_type"] == "rf":
            model = ensemble.RandomForestRegressor(random_state=0)
        elif study_parameters["model_type"] == "lasso":
            model = linear_model.Lasso(max_iter=20000)
        elif study_parameters["model_type"] == "svr":
            model = svm.SVR()
        else:
            print("What are you doing?")
            print(study_parameters["model_type"])
            
    elif study_parameters["prediction_task"] == "classification":
        if study_parameters["model_type"] == "rf":
            model = ensemble.RandomForestRegressor(random_state=0)
        elif model_type == "logistic_regression":
            study_parameters["model_type"] = linear_model.LogisticRegression()
        elif model_type == "svr":
            study_parameters["model_type"] = svm.SVM()
        else:
            print("What are you doing?")
            print(study_parameters["model_type"])
        
    if study_parameters["cross_validation_type"] == "grid":
        model = GridSearchCV(model, study_parameters["model_parameters"], n_jobs = study_parameters["n_jobs"], pre_dispatch = study_parameters["n_jobs"], cv = study_parameters["inner_loop_cv_k_folds"])
    elif study_parameters["cross_validation_type"] == "random":
        model = RandomizedSearchCV(model, study_parameters["model_parameters"], n_jobs = study_parameters["n_jobs"], pre_dispatch = study_parameters["n_jobs"], cv = study_parameters["inner_loop_cv_k_folds"], n_iter = 50, random_state = 0)
    
    return model
