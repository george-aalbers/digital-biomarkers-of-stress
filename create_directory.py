'''
Description
---
This function builds a directory for one ML/DL experiment from scratch. 

Input
---
param experiment_number: an integer signifying for which experiment we need to build a directory

Output
---
The output of this script is a new folder with the name "Experiment_n", with subfolders "data", "models", "results", "visualization"
'''

import os

def create_folder(experiment_number):

    # Get current working directory
    directory = os.getcwd() 

    # Create a new folder in the current directory
    newpath = 'experiment-' + str(experiment_number)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Change directory to new folder
    os.chdir(directory + "/" + newpath)

    for newpath in ["data","models","results","explanations","samples","visualizations"]:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
    # Change directory back to top-level folder
    os.chdir(directory)
            
def create_directory(number_of_experiments):
    
    # Loop through the number of experiments and create a folder for each
    for experiment in range(1, number_of_experiments + 1):
        create_folder(experiment)
        
    # Create folder for markdown input .csv files
    directory = os.getcwd() 

    # Create a new folder in the current directory
    newpath = 'baseline'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Create a new folder in the current directory
    newpath = 'markdown'
    if not os.path.exists(newpath):
        os.makedirs(newpath)