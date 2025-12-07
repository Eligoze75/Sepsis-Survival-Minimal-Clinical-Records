import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV
import click
import joblib
import shap
import os


PAR_PATH = os.path.dirname(os.path.dirname(__file__))
D_TRAIN_FILENAME = os.path.join(PAR_PATH, "data/processed/sepsis_train.csv")
D_TEST_FILENAME = os.path.join(PAR_PATH, "data/processed/sepsis_test.csv")
NUMERIC_FEATURES = ["age", "episode_number"]
CATEGORICAL_FEATURES = ["sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "hospital_outcome"
RANDOM_STATE = 15
MODEL_PATH = os.path.join(PAR_PATH, "results/models/logistic_reg.pkl")
CLF_METRICS_PATH = os.path.join(PAR_PATH, "results/tables/classification_metrics.csv")
CLF_TEST_PLOT = os.path.join(PAR_PATH, "results/figures/score_by_target_class.png")
CLF_COEFS_PATH = os.path.join(PAR_PATH, "results/tables/model_coefficients.csv")
CLF_SHAP_PLOT = os.path.join(PAR_PATH, "results/figures/shap_values_plot.png")


def load_data(train_filename, test_filename):
    # Read and split the data
    click.echo("[DATA COLLECTION] Reading train and test datasets...")
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)
    click.echo("[DATA COLLECTION] Split features and target...")
    X_train, y_train = (
        train_df[FEATURES],
        train_df[TARGET],
    )

    X_test, y_test = (
        test_df[FEATURES],
        test_df[TARGET],
    )
    return X_train, X_test, y_train, y_test


def model_training(X, y):
    # Create preprocessor
    ## As suggested by EDA, we standardize numeric features and one-hot encode categorical features
    click.echo("[FEATURE ENGINEERING] Creating column transformer...")
    lr_preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(drop="if_binary"), CATEGORICAL_FEATURES),
    )
    click.echo("[FEATURE ENGINEERING] Creating model pipeline...")
    # Create Pipeline
    logistic_pipe = make_pipeline(
        lr_preprocessor,
        LogisticRegression(random_state=RANDOM_STATE),
    )

    # Tune the model
    click.echo("[MODEL TUNING] RandomizedSearchCV starting...")
    param_grid = {
        "logisticregression__C": loguniform(1e-4, 1e2),
        "logisticregression__class_weight": [None, "balanced"],
        "logisticregression__max_iter": [500, 1000, 2000, 3000, 4000, 5000],
    }
    lr_random_search = RandomizedSearchCV(
        logistic_pipe,
        param_grid,
        n_iter=150,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,
        cv=5,
    )
    lr_random_search.fit(X, y)
    click.echo(
        "[MODEL TUNING] RandomizedSearchCV finished successfully -> saving optimal model"
    )
    lr_best_model = lr_random_search.best_estimator_

    joblib.dump(lr_best_model, MODEL_PATH)
    click.echo(f"Successfully saved model as: {MODEL_PATH}")
    return lr_best_model


def classification_metrics(model, X_train, X_test, y_train, y_test):
    # Classification Metrics(adapted from DSCI 573 lecture 1)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    roc_auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    classification_metrics = pd.DataFrame(
        {
            "dataset": ["train", "test"],
            "roc auc": [roc_auc_train, roc_auc_test],
            "precision": [precision_train, precision_test],
            "recall": [recall_train, recall_test],
            "F1 score": [f1_train, f1_test],
        }
    )
    click.echo(classification_metrics)
    dir_ = os.path.dirname(CLF_METRICS_PATH)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    classification_metrics.to_csv(CLF_METRICS_PATH, index=False)
    click.echo(f"Successfully saved classification metrics to: {CLF_METRICS_PATH}")


def classification_plot(clf, X, y, features):
    click.echo("[MODELING] creating classification histogram...")
    died = X.loc[y == 0]
    y_hat_died = clf.predict_proba(died[features])[:, 1]
    survived = X.loc[y == 1]
    y_hat_survived = clf.predict_proba(survived[features])[:, 1]

    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=y_hat_died,
        color="#CD7F32",
        kde=False,
        label="Died",
        alpha=0.7,
        bins=15,
    )
    sns.histplot(
        data=y_hat_survived,
        color="#4A5F80",
        kde=False,
        label="Survived",
        alpha=0.7,
        bins=15,
    )
    plt.xlabel("Probability of Survival")
    plt.legend()
    plt.tight_layout()
    plt.title("Histogram of Model Probability for Died vs Survived")
    plt.savefig(
        CLF_TEST_PLOT,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    click.echo(f"Successfully saved visualization as: {CLF_TEST_PLOT}")
    plt.show()


def model_interpretation(model, X_train, X_test):
    click.echo("[Model Interpretation] Coefficients")
    feature_names = model.named_steps["columntransformer"].get_feature_names_out()
    clean_feature_names = []
    for f in feature_names:
        clean = f.replace("onehotencoder__", "").replace("standardscaler__", "")
        # renaming sex_1
        if clean == "sex_male":
            clean_feature_names.append("is_male")
        elif clean == "sex_female":
            clean_feature_names.append("is_female")
        else:
            clean_feature_names.append(clean)

    intercept = model.named_steps["logisticregression"].intercept_
    coeffs = model.named_steps["logisticregression"].coef_
    click.echo(f"Model intercept: {intercept}")
    click.echo(f"Model coefficents:")
    df_coefs = pd.DataFrame(dict(zip(clean_feature_names, coeffs[0])))
    click.echo(df_coefs)
    df_coefs.to_csv(CLF_COEFS_PATH, index=False)
    click.echo(f"Successfully saved model coefficients to: {CLF_COEFS_PATH}")
    click.echo("[Model Interpretation] Computing SHAP Values")
    # Transform train and test sets through the preprocessor
    X_train_s = model.named_steps["columntransformer"].transform(X_train)
    X_test_s = model.named_steps["columntransformer"].transform(X_test)
    # Extract the trained LogisticRegression model
    logreg = model.named_steps["logisticregression"]
    # SHAP explainer
    explainer = shap.LinearExplainer(logreg, X_train_s)
    shap_values = explainer.shap_values(X_test_s)
    plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values, X_test_s, feature_names=clean_feature_names)
    plt.title("Logistic Classifier Shap values")
    plt.tight_layout()
    plt.savefig(
        CLF_SHAP_PLOT,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    click.echo(f"Successfully saved visualization as: {CLF_SHAP_PLOT}")
    plt.show()


@click.command()
@click.option(
    "--train_filename",
    type=str,
    default=D_TRAIN_FILENAME,
    show_default=True,
    required=False,
    help="Path to cleaned TRAIN CSV",
)
@click.option(
    "--test_filename",
    type=str,
    default=D_TEST_FILENAME,
    show_default=True,
    required=False,
    help="Path to cleaned TEST CSV",
)
@click.option(
    "--output-table",
    type=str,
    help="Path to directory where the table will be written to",
)
def main(train_filename, test_filename):
    """Reads and splits the cleaned data, fits a sepsis prediction model,
    and outputs a table summarizing the classification metrics."""

    X_train, X_test, y_train, y_test = load_data(train_filename, test_filename)
    clf = model_training(X_train, y_train)
    classification_metrics(clf, X_train, X_test, y_train, y_test)
    classification_plot(clf, X_test, y_test, FEATURES)
    model_interpretation(clf, X_train, X_test)


if __name__ == "__main__":
    main()
