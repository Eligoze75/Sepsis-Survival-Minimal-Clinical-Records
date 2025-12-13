import pandas as pd


def clean_survival_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and transform the sepsis survival dataset.

    This function renames columns, maps categorical values to human-readable
    labels, and performs basic sanity checks. It does NOT read from or write
    to disk and does NOT exit the program.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sepsis survival dataset.
    verbose : bool, default=True
        If True, print summary information during cleaning.

    Returns
    -------
    pd.DataFrame
        Cleaned sepsis survival dataset.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is empty or required columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame must not be empty")

    required_columns = {
        "age_years",
        "sex_0male_1female",
        "hospital_outcome_1alive_0dead",
    }

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    cleaned = df.copy()

    # Rename columns
    cleaned = cleaned.rename(
        columns={
            "age_years": "age",
            "sex_0male_1female": "sex",
            "hospital_outcome_1alive_0dead": "hospital_outcome",
        }
    )

    # Map categorical variables
    cleaned["sex"] = cleaned["sex"].map({0: "male", 1: "female"})
    cleaned["hospital_outcome_cat"] = cleaned["hospital_outcome"].map(
        {0: "Died", 1: "Survived"}
    )

    if verbose:
        print("[clean_survival_data] Missing values summary:")
        print(cleaned.isna().mean())

    return cleaned
