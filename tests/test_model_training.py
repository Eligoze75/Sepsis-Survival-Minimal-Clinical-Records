import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.modeling_and_evaluation import model_training


@pytest.fixture
def valid_training_data():
    """Valid and expected training input."""
    X_train = pd.DataFrame(
        {
            "age": [21, 20, 21, 77, 72],
            "sex": ["female", "female", "female", "male", "male"],
            "episode_number": [1, 1, 1, 2, 5],
        }
    )
    y_train = pd.Series([1, 0, 1, 0, 1])
    return X_train, y_train


def test_model_training_returns_fitted_pipeline(valid_training_data):
    """
    Given a valid training data when model_training is called
    then it should return a fitted sklearn Pipeline.
    """
    X, y = valid_training_data

    model = model_training(X, y)

    assert isinstance(model, Pipeline)


def test_trained_model_can_predict(valid_training_data):
    """
    Given a trained model when calling predict on valid input
    then it should return predictions of correct length.
    """
    X, y = valid_training_data
    model = model_training(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_model_training_fails_with_null_values(valid_training_data):
    """
    Given training data with null values when model_training is called
    then it should raise a ValueError.
    """
    X, y = valid_training_data
    X.loc[0, "age"] = None  # introduce null

    with pytest.raises(ValueError):
        model_training(X, y)


def test_model_training_fails_with_unvalid_category(valid_training_data):
    """
    Given training data with an unseen and unvalid category
    when model_training is called then it should raise a ValueError.
    """
    X, y = valid_training_data
    X.loc[0, "sex"] = 3  # introduce unvalid category

    with pytest.raises(ValueError):
        model_training(X, y)
