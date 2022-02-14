import shap
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def shap_values_single_model(loop_id, n_observations, study_parameters):

    # Create model id
    model_id = study_parameters["model_type"] + "_" + str(loop_id)
    
    # Load the trained model
    model = pickle.load(open(study_parameters["model_output_path"] + model_id + ".pkl", 'rb'))
    
    # Load the required feature dataframe
    X = pd.read_csv(study_parameters["data_output_path"] + "X_train_" + str(loop_id) + ".csv", index_col = 0)
    
    # Sample n_observations observations to visualize
    X_sample = shap.utils.sample(X, n_observations)

    # Sampled values to file
    X_sample = pd.DataFrame(X_sample)
    X_sample.columns = X.columns   
    
    if "X_sample_" + str(model_id) + ".csv" in os.listdir(study_parameters["data_samples_output_path"]):
    
        X_sample.to_csv(study_parameters["data_samples_output_path"] + "X_sample_" + str(model_id) + ".csv", mode = "a", header = None)
        
    else:
        
        X_sample.to_csv(study_parameters["data_samples_output_path"] + "X_sample_" + str(model_id) + ".csv")
    
    # Run KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_sample)
    
    # Get shap_values
    shap_values = explainer.shap_values(X_sample)
    shap_values = pd.DataFrame(shap_values)
    shap_values.columns = X.columns    
    
    if "shap_values_" + str(model_id) + ".csv" in os.listdir(study_parameters["explanations_output_path"]):
    
        shap_values.to_csv(study_parameters["explanations_output_path"] + "shap_values_" + str(model_id) + ".csv", mode = "a", header = None)
        
    else:
        
        shap_values.to_csv(study_parameters["explanations_output_path"] + "shap_values_" + str(model_id) + ".csv")
    
def shap_values_multiple_models():
    
    study_parameters = pd.read_json("study_parameters.json")
    
    # Nomothetic models
    for study_id in range(0,3,1):
        for loop_id in range(1,6,1):
            shap_values_single_model(loop_id, 100, study_parameters.iloc[study_id,:])
        
    # Idiographic models
    participant_list = pd.read_csv("data.csv", index_col = 0).id.unique()
    for study_id in range(3,6,1):
        for participant_id in participant_list:
            shap_values_single_model(participant_id, 100, study_parameters.iloc[study_id,:])

def rename_shap_columns():
    
    study_parameters = pd.read_json("study_parameters.json")
    
    # Nomothetic models
    for study_id in range(0,3,1):
        for loop_id in range(1,6,1):
            model_id = study_parameters["model_type"][study_id] + "_" + str(loop_id)
            X = pd.read_csv(study_parameters["data_output_path"][study_id] + "X_train_" + str(loop_id) + ".csv", index_col = 0)
            X_sample = pd.read_csv(study_parameters["data_samples_output_path"][study_id] + "X_sample_" + str(model_id) + ".csv", index_col = 0, header = None)
            X_sample.columns = X.columns
            shap_values = pd.read_csv(study_parameters["explanations_output_path"][study_id] + "shap_values_" + str(model_id) + ".csv", index_col = 0, header = None)
            shap_values.columns = X_sample.columns
            X_sample.to_csv(study_parameters["data_samples_output_path"][study_id] + "X_sample_" + str(model_id) + ".csv")
            shap_values.to_csv(study_parameters["explanations_output_path"][study_id] + "shap_values_" + str(model_id) + ".csv")
        
    # Idiographic models
    participant_list = pd.read_csv("data.csv", index_col = 0).id.unique()
    for study_id in range(3,6,1):
        for participant_id in participant_list:
            model_id = study_parameters["model_type"][study_id] + "_" + str(participant_id)            
            X = pd.read_csv(study_parameters["data_output_path"][study_id] + "X_train_" + str(participant_id) + ".csv", index_col = 0)
            X_sample = pd.read_csv(study_parameters["data_samples_output_path"][study_id] + "X_sample_" + str(model_id) + ".csv", index_col = 0, header = None)
            X_sample.columns = X.columns
            shap_values = pd.read_csv(study_parameters["explanations_output_path"][study_id] + "shap_values_" + str(model_id) + ".csv", index_col = 0, header = None)
            shap_values.columns = X_sample.columns
            X_sample.to_csv(study_parameters["data_samples_output_path"][study_id] + "X_sample_" + str(model_id) + ".csv")
            shap_values.to_csv(study_parameters["explanations_output_path"][study_id] + "shap_values_" + str(model_id) + ".csv")
            
def beeswarm(shap_values, X_sample, model_id, study_parameters):
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(study_parameters["visualizations_output_path"] + "beeswarm_plot_" + model_id + ".png", dpi=150, bbox_inches='tight')
    plt.clf()
            
def feature_importance_nomothetic_models():

    root = os.getcwd()
    
    dfs = pd.DataFrame()
    
    for i in range(1,4,1):

        folder = root + "/experiment-" + str(i) + "/explanations/"
        
        files = pd.Series(os.listdir(folder))
        
        files = folder + files
        
        files = files[files.str[-4:] == ".csv"]
        
        for file in files:
            
            df = pd.read_csv(file, index_col = 0)
            
            model = file[64:-4].split("_")[0]
            fold = file[64:-4].split("_")[1]
            
            df["model"] = np.repeat(model, df.shape[0])
            df["fold"] = np.repeat(fold, df.shape[0])
            
            dfs = pd.concat([dfs, df], axis = 0)

    dfs.set_index(["model", "fold"], inplace = True)
            
    dfs = dfs.abs().groupby('model').mean()        
            
    dfs.to_csv("feature_importance_nomothetic_models.csv")

def feature_importance_idiographic_models():

    root = os.getcwd()
    
    dfs = pd.DataFrame()
    
    for i in range(4,7,1):

        folder = root + "/experiment-" + str(i) + "/explanations/"
        
        files = pd.Series(os.listdir(folder))
        
        files = folder + files
        
        files = files[files.str[-4:] == ".csv"]

        for file in files:

            df = pd.read_csv(file, index_col = 0)
            
            model = file[64:-4].split("_")[0]
            participant = file[64:-4].split("_")[1]
            
            df["model"] = np.repeat(model, df.shape[0])
            df["id"] = np.repeat(participant, df.shape[0])
            
            dfs = pd.concat([dfs, df], axis = 0)
    
    dfs.set_index(["model", "id"], inplace = True)
    
    dfs = dfs.abs().groupby('model').mean()
    
    dfs.to_csv("feature_importance_idiographic_models.csv")

def correlations_between_features_and_shapley_values(X, shap_values):
    
    # Dataframe for saving results
    df = pd.DataFrame()
    
    #calculate Spearman Rank correlation and corresponding p-value
    for i in range(X.shape[1]):
        rho, p = spearmanr(X.iloc[:,i], shap_values.iloc[:,i])
        
        df = pd.concat([df, pd.DataFrame({"rho":rho, "p":p}, index=[i])])
        
    return df

def get_correlations_in_right_format(X, shap_values, participant):
    correlations = correlations_between_features_and_shapley_values(X, shap_values)
    correlations.index = X.columns.tolist()
    correlations.reset_index(inplace=True)
    correlations.columns = ["feature", "rho", "p"]
    correlations.index = np.repeat(participant, X.shape[1])
    return correlations.reset_index()

def correlations_between_features_and_shapley_values_multiple_participants():
    
    participant_list = pd.read_csv("data.csv").id.unique().tolist()
    
    model = ["lasso_", "svr_", "rf_"]
    
    root = os.getcwd()
    
    for i in range(4,7,1):

        for participant in participant_list:

            # Select model path
            shap_values = pd.read_csv(root + "/experiment-" + str(i) + "/explanations/shap_values_" + model[i-4] + participant + ".csv", index_col = 0)
            
            # Select data path
            X = pd.read_csv(root + "/experiment-" + str(i) + "/samples/X_sample_" + model[i-4] + participant + ".csv", index_col = 0)

            # Get correlations in right format
            correlations = get_correlations_in_right_format(X, shap_values, participant)
            
            # Rename index to model name for easy postprocessing
            correlations.index = np.repeat(model[i-4][:-1], X.shape[1])
            
            # Write to file
            if "shapley_correlations.csv" in os.listdir():
                print("You are now adding rows to a file. Are you sure?")
                correlations.to_csv("shapley_correlations.csv", mode = 'a', header = None)
            else:
                correlations.to_csv("shapley_correlations.csv")
                
def select_features(df):
    return df[df.p < 0.05]

def calculate_proportion_positive(df):
    return sum(df.rho > 0)/224

def calculate_proportion_negative(df):
    return sum(df.rho < 0)/224

def calculate_proportion_zero(df):
    return sum(df.rho == 0)/224

def concatenate_proportions(df):

    study_parameters = pd.read_json("study_parameters.json")
    
    positive = df.groupby(['model', 'feature']).apply(calculate_proportion_positive).reset_index()
    negative = df.groupby(['model', 'feature']).apply(calculate_proportion_negative).reset_index()
    zero = df.groupby(['model', 'feature']).apply(calculate_proportion_zero).reset_index()
    
    proportions = pd.merge(positive, negative, on = ["model", "feature"], how = "outer")
    proportions = pd.merge(proportions, zero, on = ["model", "feature"], how = "outer")
    proportions.columns = ["model", "feature", "positive", "negative", "zero"]
    
    return proportions

def bar_plot_per_model(df):

    from matplotlib import pyplot as plt

    model_type = "I-" + df.model.unique()[0]
    
    if model_type == "I-svr":

        plt.rcParams["figure.figsize"] = [5, 10]
        plt.rcParams["figure.autolayout"] = True
        
        b1 = plt.barh(df["feature"], df["positive"], color="#0589fe")

        b2 = plt.barh(df["feature"], df["negative"], color="#fe054e")

        plt.title(model_type.upper())

        plt.xlim(0,0.6)

        plt.xlabel("Proportion of participants with positive or negative correlation")
        
        plt.yticks(ticks=[],labels=[])
        
        root = os.getcwd()

        plt.tight_layout()
        
        plt.savefig(root + "/markdown/barplot-" + model_type + ".png")

        plt.clf()
        
    elif model_type == "I-lasso":

        plt.rcParams["figure.figsize"] = [5, 10]
        plt.rcParams["figure.autolayout"] = True
        
        b1 = plt.barh(df["feature"], df["positive"], color="#0589fe")

        b2 = plt.barh(df["feature"], df["negative"], color="#fe054e")

        plt.title(model_type.upper())
        
        plt.yticks(ticks=[],labels=[])
        
        plt.xlim(0,0.6)

        root = os.getcwd()
        
        plt.tight_layout()
        
        plt.savefig(root + "/markdown/barplot-" + model_type + ".png")

        plt.clf()
        
    else:

        plt.rcParams["figure.figsize"] = [5, 10]
        plt.rcParams["figure.autolayout"] = True
        
        b1 = plt.barh(df["feature"], df["positive"], color="#0589fe")

        b2 = plt.barh(df["feature"], df["negative"], color="#fe054e")

        plt.legend([b1, b2], ["Positive", "Negative"], title="Relationship", loc="upper right")

        plt.title(model_type.upper())

        plt.xlim(0,0.6)
        
        plt.yticks(ticks=[],labels=[])
        
        root = os.getcwd()

        plt.tight_layout()
        
        plt.savefig(root + "/markdown/barplot-" + model_type + ".png")

        plt.clf()

def get_legend(df):

    plt.rcParams["figure.figsize"] = [5, 10]
    plt.rcParams["figure.autolayout"] = True

    b1 = plt.barh(df["feature"], df["positive"], color="#0589fe")

    b2 = plt.barh(df["feature"], df["negative"], color="#fe054e")

    plt.legend([b1, b2], ["Positive", "Negative"], title="Relationship", loc="upper right")

    plt.title("Legend")

    plt.xlim(0,0.6)

    plt.xticks(ticks=[],labels=[])
    
    root = os.getcwd()
    
    plt.tight_layout()
    
    plt.savefig(root + "/markdown/barplot_legend.png")

    plt.clf()
        
def get_input_bar_plots():
    correlations_between_features_and_shapley_values_multiple_participants()
    study_parameters = pd.read_json("study_parameters.json")
    df = pd.read_csv("shapley_correlations.csv")
    df.columns = ["model", "participant", "feature", "rho", "p"]
    df.rho = df.rho.fillna(0)
    df.p = df.p.fillna(1)
    df.rho[df.p > 0.05] = 0
    proportions = concatenate_proportions(df)
    proportions.model.replace({"lasso": "I-LASSO", "svr": "I-SVR", "rf": "I-RF"}, inplace = True)
    proportions.to_csv(study_parameters["markdown_path"][0] + "barplot_input.csv")

import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

def get_ranking(df):
    df.reset_index(inplace=True, drop=True)
    df = df.reset_index().rename({"index":"ranking"}, axis = 1)
    return df[["model","feature","ranking"]]
    
def beeswarm(shap_values, X_sample, study_parameters, model_id):
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(study_parameters["markdown_path"][0] + "beeswarm_plot_" + str(model_id) + ".png", dpi=150, bbox_inches='tight')
    plt.clf()

def correlations_between_features_and_shapley_values(X, shap_values):
    
    # Dataframe for saving results
    df = pd.DataFrame()
    
    #calculate Spearman Rank correlation and corresponding p-value
    for i in range(X.shape[1]):
        rho, p = spearmanr(X.iloc[:,i], shap_values.iloc[:,i])
        
        df = pd.concat([df, pd.DataFrame({"rho":rho, "p":p}, index=[i])])
        
    return df

def get_correlations_in_right_format(X, shap_values, participant):
    correlations = correlations_between_features_and_shapley_values(X, shap_values)
    correlations.index = X.columns.tolist()
    correlations.reset_index(inplace=True)
    correlations.columns = ["feature", "rho", "p"]
    correlations.index = np.repeat(participant, X.shape[1])
    return correlations.reset_index()

def correlations_between_features_and_shapley_values_multiple_participants():
    
    participant_list = pd.read_csv("data.csv").id.unique().tolist()
    
    model = ["lasso_", "svr_", "rf_"]
    
    root = os.getcwd()
    
    for i in range(4,7,1):

        for participant in participant_list:

            # Select model path
            shap_values = pd.read_csv(root + "/experiment-" + str(i) + "/explanations/shap_values_" + model[i-4] + participant + ".csv", index_col = 0)
            
            # Select data path
            X = pd.read_csv(root + "/experiment-" + str(i) + "/samples/X_sample_" + model[i-4] + participant + ".csv", index_col = 0)

            # Get correlations in right format
            correlations = get_correlations_in_right_format(X, shap_values, participant)
            
            # Rename index to model name for easy postprocessing
            correlations.index = np.repeat(model[i-4][:-1], X.shape[1])
            
            # Write to file
            if "shapley_correlations.csv" in os.listdir():
                correlations.to_csv("shapley_correlations.csv", mode = 'a', header = None)
            else:
                correlations.to_csv("shapley_correlations.csv")

def figure_1():

    study_parameters = pd.read_json("study_parameters.json")
    t4 = pd.read_csv(study_parameters["markdown_path"][0] + "table-4.csv")
    t4.columns = ["model", "feature", "importance"]
    t5 = pd.read_csv(study_parameters["markdown_path"][0] + "table-5.csv", index_col = 0)
    
    feature_ranking_nomothetic_models = t4.groupby("model").apply(get_ranking).pivot(index=["feature"], columns = ["model"], values = ["ranking"]) + 1
    feature_ranking_idiographic_models = t5.groupby("model").apply(get_ranking).pivot(index=["feature"], columns = ["model"], values = ["ranking"]) + 1
    feature_ranking_nomothetic_models["median_rank_nomothetic_models"] = feature_ranking_nomothetic_models.median(axis=1).astype(int)
    feature_ranking_idiographic_models["median_rank_idiographic_models"] = feature_ranking_idiographic_models.median(axis=1).astype(int)
    feature_ranking = pd.merge(feature_ranking_nomothetic_models, feature_ranking_idiographic_models, on = "feature", how = "outer")
    feature_ranking = feature_ranking.fillna(34).astype(int)
    feature_ranking = feature_ranking.sort_values(by = "median_rank_nomothetic_models")
    feature_ranking = feature_ranking.droplevel(axis = 1, level = 0)
    feature_ranking.columns = ["N-LASSO", "N-RF", "N-SVR", "N-median", "I-LASSO", "I-RF", "I-SVR", "I-median"]

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(feature_ranking.iloc[:11,:], annot=True, fmt="d", linewidths=.5, cmap="Blues_r", cbar = False)
    ax.set(ylabel="")
    ax.set_xticklabels(labels = ["N-LASSO", "N-RF", "N-SVR", "N-median", "I-LASSO", "I-RF", "I-SVR", "I-median"], rotation = 30)
    ax.xaxis.set_tick_params(labeltop='on')
    plt.tight_layout()
    plt.savefig(study_parameters["markdown_path"][0] + "feature_importance.png")
    plt.clf()

def figure_2():
    study_parameters = pd.read_json("study_parameters.json")
    X_sample = pd.read_csv(study_parameters["data_samples_output_path"][2] + "X_sample_rf_1.csv")
    shap_values = np.array(pd.read_csv(study_parameters["explanations_output_path"][2] + "shap_values_rf_1.csv"))
    beeswarm(shap_values, X_sample, study_parameters, "rf_1")
    plt.clf()
    
def figure_3():

    import matplotlib
    import seaborn as sns

    get_input_bar_plots()
    
    study_parameters = pd.read_json("study_parameters.json")
    df = pd.read_csv(study_parameters["markdown_path"][0] + "barplot_input.csv")
    
    sns.set_theme(style="white")
    g = sns.FacetGrid(df, col="model", col_order = ["I-LASSO", "I-SVR", "I-RF"], xlim=[0,1], sharex=True, sharey=True, height=10, aspect=0.45, margin_titles=True)
    ax1 = g.map(sns.barplot, "positive", "feature", color = "#0589fe", orient = "h", order = df.feature.unique())
    ax1.set(xlabel='', ylabel='')
    ax2 = g.map(sns.barplot, "negative", "feature", color = "#fe054e", orient = "h", order = df.feature.unique())
    ax2.set(xlabel='', ylabel='')
    axes = g.axes.flatten()
    axes[0].set_title("I-LASSO")
    axes[1].set_title("I-SVR")
    axes[2].set_title("I-RF")
    g.fig.text(x = 0.35, y = 0.0075, s = "Proportion of participants with positive and negative correlation")

    name_to_color = {
        'Positive':   '#0589fe',
        'Negative':   '#fe054e',
    }

    patches = [matplotlib.patches.Patch(color=v, label=k) for k,v in name_to_color.items()]
    matplotlib.pyplot.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc='upper right')

    plt.savefig(study_parameters["markdown_path"][0] + "bar_plots.png")
    plt.clf()