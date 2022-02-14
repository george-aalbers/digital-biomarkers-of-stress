# Import required packages
import pandas as pd
import argparse
from explain_model import shap_values_single_model

# Read participant_list and study parameters
data = pd.read_csv("data.csv", index_col = 0)
participant_list = data.id.unique().tolist()
study_parameters = pd.read_json("study_parameters.json")

# Create argparse arguments
parser = argparse.ArgumentParser(description='Loop through participant numbers.')
parser.add_argument('study_i', metavar='', type=int, nargs='+', help='the initial parameters')
parser.add_argument('model_i', metavar='', type=int, nargs='+', help='the initial parameters')
args = parser.parse_args()

# Get Shap values, visualisations, and X sample
shap_values_single_model(participant_list[args.model_i[0]], 10, study_parameters.iloc[args.study_i[0],:])