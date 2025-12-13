import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_transformation import _clean_survival_df


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "age_years": [65, 70],
        "sex_0male_1female": [0, 1],
        "hospital_outcome_1alive_0dead": [1, 0],
    })


# expected use cases
def test_clean_survival_data_success(valid_df):
    cleaned = _clean_survival_df(valid_df, verbose=False)

    # column renaming
    assert "age" in cleaned.columns
    assert "sex" in cleaned.columns
    assert "hospital_outcome" in cleaned.columns
    assert "hospital_outcome_cat" in cleaned.columns

    # old columns removed
    assert "age_years" not in cleaned.columns
    assert "sex_0male_1female" not in cleaned.columns
    assert "hospital_outcome_1alive_0dead" not in cleaned.columns

    # value mapping
    assert cleaned["sex"].tolist() == ["male", "female"]
    assert cleaned["hospital_outcome_cat"].tolist() == ["Survived", "Died"]


# Edge cases

def test_clean_survival_data_missing_values(valid_df):
    df_with_na = valid_df.copy()
    df_with_na.loc[0, "age_years"] = None

    cleaned = _clean_survival_df(df_with_na, verbose=False)

    assert cleaned["age"].isna().sum() == 1


def test_clean_survival_data_single_row():
    df = pd.DataFrame({
        "age_years": [80],
        "sex_0male_1female": [1],
        "hospital_outcome_1alive_0dead": [1],
    })

    cleaned = _clean_survival_df(df, verbose=False)

    assert cleaned.shape[0] == 1
    assert cleaned["sex"].iloc[0] == "female"
    assert cleaned["hospital_outcome_cat"].iloc[0] == "Survived"

# error cases

def test_clean_survival_data_wrong_type():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        _clean_survival_df("not a dataframe")
