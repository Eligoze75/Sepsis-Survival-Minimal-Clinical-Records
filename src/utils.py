import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import zipfile
import io
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
#import shap

# Importing Data
def load_ucisepsis(inner_filename):
    """
    Downloads and extracts the Sepsis Survival Minimal Clinical Records dataset
    from the UCI Machine Learning Repository.

    Parameters:
        inner_filename (str): The CSV file to load from inside the inner ZIP.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    
    # URL of the outer zip
    url = (
        "https://archive.ics.uci.edu/static/public/827/"
        "sepsis%2Bsurvival%2Bminimal%2Bclinical%2Brecords.zip"
    )

    # Download outer zip
    r = requests.get(url)
    outer_zip = zipfile.ZipFile(io.BytesIO(r.content))

    # Extract inner zip
    inner_zip_name = outer_zip.namelist()[0]
    inner_zip_bytes = outer_zip.read(inner_zip_name)
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_zip_bytes))

    # Safety check: ensure the requested file exists
    if inner_filename not in inner_zip.namelist():
        raise ValueError(
            f"File '{inner_filename}' not found.\n"
            f"Available files: {inner_zip.namelist()}"
        )

    # Load CSV
    with inner_zip.open(inner_filename) as f:
        return pd.read_csv(f)
    
# EDA
def plot_bivariates(df, var, y, figsize=(10, 5)):
    """
    Plots the relationship between a feature and a binary target variable.
    Handles both continuous and categorical variables. For continuous
    features, it applies quantile binning. For high cardinality categorical
    variables, it normalizes categories. Displays application counts as bars
    and event rate as a line.

    Parameters:
    df: DataFrame containing the input data
    var: Name of the feature column to analyze
    y: Name of the binary target variable
    figsize: Tuple that defines the size of the plot. Default is (10, 5)

    Output:
    A matplotlib plot displaying volume (bar) and event rate (line) per bin
    or category of the feature.
    """
    df_ = df.copy()
    df_[var] = pd.qcut(df[var], 5, duplicates="drop")
    by_var = df_.groupby(var).agg({y: [np.size, np.mean]})[y]

    Y1 = by_var["size"]
    Y2 = by_var["mean"]
    Y_mean = np.ones(shape=(len(Y1.index))) * df_[y].mean()
    index = np.arange(len(Y1.index))
    pcts = np.arange(0.0, 1.1, 0.1)
    fig = plt.figure(figsize=figsize)
    plt.bar(index, Y1, alpha=0.3, color="gray")
    plt.grid(False)
    plt.ylabel("# applications")
    plt.twinx()
    plt.gca().set_xticks(index)
    plt.gca().set_xticklabels([Y1.index[i] for i in index], rotation=40)
    plt.plot(index, Y_mean, label=f"overall avg {y} rate", color="#1F75FE")
    plt.plot(index, Y2, marker="o", label=f"{y} rate", color="#E62020")
    plt.gca().set_yticks(pcts)
    plt.gca().set_yticklabels([" {:.2f}%".format(i * 100) for i in pcts])
    plt.grid(False)
    plt.title("{}".format(var))
    plt.ylabel("tx. evento")
    plt.xlabel(var)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.35))
    plt.show()


# MODEL EXPLAINABILITY


def get_shaps(model, x_s, x_features, model_tag, max_display=100):
    """
    Computes and visualizes SHAP values for a tree-based model using the
    SHAP library. Prints a summary plot of feature importances and returns
    the raw SHAP values.

    Parameters:
    model: Trained tree-based model compatible with SHAP (e.g., XGBoost,
    LightGBM)
    x_s: Transformed input features (e.g., from a pipeline or encoder)
    x_features: List of original feature names used for labeling
    model_tag: String label to identify the model in the printed output
    max_display: Maximum number of features to display in the summary plot
    (default is 100)

    Output:
    shap_values: Array of SHAP values for each sample and feature
    """
    print("\n", f"SHAP values for {model_tag}".center(40))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_s)
    shap.summary_plot(
        shap_values, x_s, feature_names=x_features, max_display=max_display
    )
    return shap_values


def get_important_features_important_than(shap_values, features):
    # Compute mean absolute SHAP values for each feature
    shap_importances = np.mean(np.abs(shap_values), axis=0)

    # Create a list of tuples (feature name, importance)
    feature_importance = list(zip(features, shap_importances))

    # Sort the list by importance
    sorted_feature_importance = sorted(
        feature_importance, key=lambda x: x[1], reverse=True
    )

    # Print sorted feature importance
    feature_importance_dict = {}
    i = 1
    for feature, importance in sorted_feature_importance:
        feature_importance_dict[f"{feature}"] = i
        i += 1
    rand_min_importance = np.min(
        [feature_importance_dict["RANDOM_1"], feature_importance_dict["RANDOM_2"]]
    )
    important_enough_features = sorted(
        [
            v[0]
            for v in list(feature_importance_dict.items())
            if v[1] < rand_min_importance
        ]
    )
    return important_enough_features
