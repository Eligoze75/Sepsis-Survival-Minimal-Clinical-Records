import click
import pandas as pd
import sys
import os

PAR_PATH = os.path.dirname(os.path.dirname(__file__))
DEFAULT_RAW_DATA_PATH = os.path.join(PAR_PATH, "data/raw")
DEFAULT_PROCESSED_DATA_PATH = os.path.join(PAR_PATH, "data/processed")

D_RAW_TRAIN_FILENAME = os.path.join(
    DEFAULT_RAW_DATA_PATH, "s41598-020-73558-3_sepsis_survival_primary_cohort.csv"
)
D_RAW_TEST_FILENAME = os.path.join(
    DEFAULT_RAW_DATA_PATH, "s41598-020-73558-3_sepsis_survival_study_cohort.csv"
)

D_PROCESSED_TRAIN_FILENAME = os.path.join(
    DEFAULT_PROCESSED_DATA_PATH, "sepsis_train.csv"
)
D_PROCESSED_TEST_FILENAME = os.path.join(
    DEFAULT_PROCESSED_DATA_PATH, "sepsis_test.csv"
)


def _clean_survival_df(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Core data cleaning logic for sepsis survival data.
    This function is intentionally separated from the CLI so it can be unit tested
    without invoking Click or file I/O.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    rename_map = {
        "age_years": "age",
        "sex_0male_1female": "sex",
        "hospital_outcome_1alive_0dead": "hospital_outcome",
    }

    df = df.rename(columns=rename_map)

    if "sex" in df.columns:
        df["sex"] = df["sex"].map({0: "male", 1: "female"})

    if "hospital_outcome" in df.columns:
        df["hospital_outcome_cat"] = df["hospital_outcome"].map(
            {0: "Died", 1: "Survived"}
        )

    if verbose:
        print(df.isna().mean())

    return df


@click.command()
@click.option(
    "--input-train",
    "-i",
    default=D_RAW_TRAIN_FILENAME,
    show_default=True,
    help="Path to raw TRAIN CSV",
)
@click.option(
    "--input-test",
    "-j",
    default=D_RAW_TEST_FILENAME,
    show_default=True,
    help="Path to raw TEST CSV",
)
@click.option(
    "--output-train",
    "-o1",
    default=D_PROCESSED_TRAIN_FILENAME,
    show_default=True,
    help="Path to cleaned TRAIN CSV",
)
@click.option(
    "--output-test",
    "-o2",
    default=D_PROCESSED_TEST_FILENAME,
    show_default=True,
    help="Path to cleaned TEST CSV",
)
def clean_data(input_train, input_test, output_train, output_test):
    """
    CLI entry point for cleaning sepsis survival datasets.
    """

    # loading input files
    try:
        train_df = pd.read_csv(input_train)
        test_df = pd.read_csv(input_test)
    except Exception as e:
        print("Error reading input files:", e)
        sys.exit(1)

    # applying cleaning logic
    train_df = _clean_survival_df(train_df)
    test_df = _clean_survival_df(test_df)

    # ensuring output directories exist
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)

    # saving outputs
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)


if __name__ == "__main__":
    clean_data()
