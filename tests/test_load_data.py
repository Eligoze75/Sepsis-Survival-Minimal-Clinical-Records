import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.modeling_and_evaluation as me

# Expected inputs
@pytest.fixture
def sample_train_dataframe():
    return pd.DataFrame({
        "age": [10, 20, 30],
        "sex": ["male", "female", "male"],
        "episode_number": [1, 2, 3],
        "hospital_outcome": [0, 1, 0],
    })

@pytest.fixture
def sample_test_dataframe():
    return pd.DataFrame({
        "age": [40, 50],
        "sex": ["female", "male"],
        "episode_number": [4, 5],
        "hospital_outcome": [1, 0],
    })

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path


# 1. Test for edge cases & error handling - Train and test dfs
def test_load_data_working_path(
    temp_directory, sample_train_dataframe, sample_test_dataframe, monkeypatch
):
    monkeypatch.setattr(me, "FEATURES", ["age", "sex", "episode_number"])
    monkeypatch.setattr(me, "TARGET", "hospital_outcome")


    train_path = temp_directory / "train.csv"
    test_path = temp_directory / "test.csv"
    sample_train_dataframe.to_csv(train_path, index=False)
    sample_test_dataframe.to_csv(test_path, index=False)

    # Act
    X_train, X_test, y_train, y_test = me.load_data(train_path, test_path)

    # Assert
    assert list(X_train.columns) == ["age", "sex", "episode_number"]
    assert list(X_test.columns) == ["age", "sex", "episode_number"]
    assert y_train.name == "hospital_outcome"
    assert y_test.name == "hospital_outcome"
    assert X_train.shape == (3, 3)
    assert X_test.shape == (2, 3)

# 2. Test for edge cases & error handling - missing train files
def test_load_data_missing_train_file_raises(temp_directory, monkeypatch):
    monkeypatch.setattr(me, "FEATURES", ["x"])
    monkeypatch.setattr(me, "TARGET", "hospital_outcome")

    test_path = temp_directory / "test.csv"
    pd.DataFrame({"x": [1], "hospital_outcome": [0]}).to_csv(test_path, index=False)

    with pytest.raises(FileNotFoundError):
        me.load_data(temp_directory / "does_not_exist.csv", test_path)

# 3. Test for edge cases & error handling - empty data frames
def test_load_data_empty_csv(temp_directory, monkeypatch):
    monkeypatch.setattr(me, "FEATURES", ["age", "sex", "episode_number"])
    monkeypatch.setattr(me, "TARGET", "hospital_outcome")

    train_path = temp_directory / "train.csv"
    test_path = temp_directory / "test.csv"

    pd.DataFrame(columns=["age", "sex", "episode_number", "hospital_outcome"]).to_csv(train_path, index=False)
    pd.DataFrame(columns=["age", "sex", "episode_number", "hospital_outcome"]).to_csv(test_path, index=False)

    X_train, X_test, y_train, y_test = me.load_data(train_path, test_path)

    assert X_train.empty
    assert y_train.empty

# 4. Test for edge cases & error handling - extra columns
def test_load_data_ignores_extra_columns(temp_directory, monkeypatch):
    monkeypatch.setattr(me, "FEATURES", ["age"])
    monkeypatch.setattr(me, "TARGET", "hospital_outcome")

    df = pd.DataFrame({
        "age": [10],
        "hospital_outcome": [1],
        "bmi": [22],
    })

    path = temp_directory / "data.csv"
    df.to_csv(path, index=False)

    X_train, X_test, y_train, y_test = me.load_data(path, path)

    assert "bmi" not in X_train.columns

# 5. Test for edge cases & error handling - missing column names
def test_load_data_missing_columns_raises(temp_directory, monkeypatch):
    monkeypatch.setattr(me, "FEATURES", ["age", "sex", "episode_number"])
    monkeypatch.setattr(me, "TARGET", "hospital_outcome")

    train_path = temp_directory / "train.csv"
    test_path = temp_directory / "test.csv"

    # Missing episode_number in train
    pd.DataFrame({
        "age": [10],
        "sex": ["male"],
        "hospital_outcome": [0],
    }).to_csv(train_path, index=False)

    pd.DataFrame({
        "age": [20],
        "sex": ["female"],
        "episode_number": [1],
        "hospital_outcome": [1],
    }).to_csv(test_path, index=False)

    with pytest.raises(KeyError):
        me.load_data(train_path, test_path)