import pandas as pd
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
import os


NUMERIC_FEATURES = ["age", "episode_number"]
CATEGORICAL_FEATURES = ["sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "hospital_outcome"
RANDOM_STATE = 15
PAR_PATH = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PAR_PATH, "results/models/logistic_reg.pkl")
CLF_METRICS_PATH = os.path.join(PAR_PATH, "results/tables/classification_metrics.csv")


@click.command()
@click.option("--train-df", type=str, required=True, help="Path to cleaned TRAIN CSV")
@click.option("--test-df", type=str, required=True, help="Path to cleaned TEST CSV")
@click.option(
    "--output-table",
    type=str,
    help="Path to directory where the table will be written to",
)
def main(train_df, test_df, output_table):
    """Reads and splits the cleaned data, fits a sepsis prediction model, and outputs a table summarizing the classification metrics."""

    # Read and split the data
    click.echo("[DATA COLLECTION] Reading train and test datasets...")
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    click.echo("[DATA COLLECTION] Split features and target...")
    X_train, y_train = (
        train_df[FEATURES],
        train_df[TARGET],
    )

    X_test, y_test = (
        test_df[FEATURES],
        test_df[TARGET],
    )

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
    lr_random_search.fit(X_train, y_train)
    click.echo(
        "[MODEL TUNING] RandomizedSearchCV finished successfully -> saving optimal model"
    )
    lr_best_model = lr_random_search.best_estimator_

    joblib.dump(lr_best_model, MODEL_PATH)
    click.echo(f"Successfully saved model as: {MODEL_PATH}")
    # Classification Metrics(adapted from DSCI 573 lecture 1)
    y_pred_train = lr_best_model.predict(X_train)
    y_pred_test = lr_best_model.predict(X_test)
    roc_auc_train = roc_auc_score(y_train, lr_best_model.predict_proba(X_train)[:, 1])
    roc_auc_test = roc_auc_score(y_test, lr_best_model.predict_proba(X_test)[:, 1])
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
    click.echo(f"Successfully saved training summary stats to: {CLF_METRICS_PATH}")


if __name__ == "__main__":
    main()
