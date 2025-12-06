# sepsis_modeling_and_evaluation.py
# date: 2025-12-05
import click
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

@click.command()
@click.option("--train-df", type=str, required=True, help="Path to cleaned TRAIN CSV")
@click.option("--test-df", type=str, required=True, help="Path to cleaned TEST CSV")
@click.option('--table-to', type=str, help="Path to directory where the table will be written to")
@click.option('--model-to', type=str, help="Path to directory where the model object will be written to")

def main(train_df, test_df, model_to, table_to):
    '''Reads and splits the cleaned data, fits a sepsis prediction model, and outputs a table summarizing the classification metrics.'''

    RANDOM_STATE = 15
    
    # Read and split the data
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    
    X_train, y_train = (
        train_df.drop(columns=["hospital_outcome", "hospital_outcome_cat"]),
        train_df["hospital_outcome"],
    )

    X_test, y_test = (
        test_df.drop(columns=["hospital_outcome", "hospital_outcome_cat"]),
        test_df["hospital_outcome"],
    )

    # Create preprocessor
    numeric_features = ["age", "episode_number"]
    categorical_features = ["sex"]
    features = numeric_features + categorical_features

    # As suggested by EDA, we standardize numeric features and one-hot encode categorical features
    lr_preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop="if_binary"), categorical_features),
    )
    
    # Create Pipeline
    logistic_pipe = make_pipeline(
        lr_preprocessor,
        LogisticRegression(random_state=RANDOM_STATE),
    )
    
    # Tune the model
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
        cv=5
    )
    lr_random_search.fit(X_train, y_train)
    lr_best_model = lr_random_search.best_estimator_   

    with open("lr_fit.pickle", 'wb') as f:
    pickle.dump(lr_fit, f)
    
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
       
    with open("classification_metrics.pickle", 'wb') as f:
    pickle.dump(classification_metrics, f)

if __name__ == "__main__":
    main()