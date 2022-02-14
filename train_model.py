'''
Description
---
This function trains models and saves them in a pickle file. 

Input
---

param model: initialized GridSearch or RandomSearchCV object
param study_parameters: .csv specifying where to save pickled model
param loop_id: value representing either the iteration number for the nomothetic model training loop or the participant ID for the idiographic model training loop

Output
---
The output of this script is a pickled model.
'''

# Imports
import pandas as pd
from build_model import build_model
import pickle

def train_model(model, X, y, study_parameters, loop_id):

    # Fit model
    model.fit(X, y)

    # Write model to file
    pickle.dump(model, open(study_parameters["model_output_path"] + study_parameters["model_type"] + '_' + str(loop_id) + '.pkl','wb'))
    
    return model