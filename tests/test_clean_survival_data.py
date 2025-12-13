import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.clean_data import clean_survival_data


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "age_years": [65, 70],
            "sex_0male_1female": [0, 1],
            "hospital_outcome_1alive_0dead": [1, 0],
        }
    )

# Expected use cases

def test_clean_survival_data_success(valid_df):
    cleaned = clean_survival_data(valid_df, verbose=False)

    assert isinstance(cleaned, pd.DataFrame)
    assert "age" in cleaned.columns
    assert "sex" in cleaned.columns
    assert "hospital_outcome_cat" in cleaned.columns
    assert cleaned.loc[0, "sex"] == "male"
    assert cleaned.loc[1, "hospital_outcome_cat"] == "Died"


# Edge cases

def test_clean_survival_data_missing_values(valid_df):
    valid_df.loc[0, "age_years"] = None
    cleaned = clean_survival_data(valid_df, verbose=False)

    assert cleaned["age"].isna().sum() == 1


def test_clean_survival_data_empty_dataframe():
    with pytest.raises(ValueError, match="must not be empty"):
        clean_survival_data(pd.DataFrame(), verbose=False)


# Error cases

def test_clean_survival_data_wrong_type():
    with pytest.raises(TypeError):
        clean_survival_data([1, 2, 3], verbose=False)


def test_clean_survival_data_missing_columns(valid_df):
    df = valid_df.drop(columns=["age_years"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_survival_data(df, verbose=False)
