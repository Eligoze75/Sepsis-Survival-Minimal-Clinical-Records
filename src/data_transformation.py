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
D_PROCESSED_TEST_FILENAME = os.path.join(DEFAULT_PROCESSED_DATA_PATH, "sepsis_test.csv")


@click.command()
@click.option(
    "--input-train",
    "-i",
    default=D_RAW_TRAIN_FILENAME,
    required=False,
    show_default=True,
    help="Path to raw TRAIN CSV",
)
@click.option(
    "--input-test",
    "-j",
    default=D_RAW_TEST_FILENAME,
    required=False,
    show_default=True,
    help="Path to raw TEST CSV",
)
@click.option(
    "--output-train",
    "-o1",
    default=D_PROCESSED_TRAIN_FILENAME,
    required=False,
    show_default=True,
    help="Path to cleaned TRAIN CSV",
)
@click.option(
    "--output-test",
    "-o2",
    default=D_PROCESSED_TEST_FILENAME,
    required=False,
    show_default=True,
    help="Path to cleaned TEST CSV",
)
def clean_data(input_train, input_test, output_train, output_test):
    """Clean and transform train and test datasets."""

    # Load input files
    click.echo("[Loading data] Reading csv files")
    try:
        train_df = pd.read_csv(input_train)
        test_df = pd.read_csv(input_test)
    except Exception as e:
        print("Error reading input files:", e)
        sys.exit(1)

    # Rename columns
    click.echo("[Preprocessing] Renaming columns")
    rename_map = {
        "age_years": "age",
        "sex_0male_1female": "sex",
        "hospital_outcome_1alive_0dead": "hospital_outcome",
    }
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    # Map sex and outcome
    click.echo("[Preprocessing] Renaming Sex categories for better interpretability")
    train_df["sex"] = train_df["sex"].map({0: "male", 1: "female"})
    test_df["sex"] = test_df["sex"].map({0: "male", 1: "female"})

    click.echo("[Preprocessing] Renaming Target categories for better interpretability")
    outcome_map = {0: "Died", 1: "Survived"}
    train_df["hospital_outcome_cat"] = train_df["hospital_outcome"].map(outcome_map)
    test_df["hospital_outcome_cat"] = test_df["hospital_outcome"].map(outcome_map)

    # Report missing values
    click.echo("[Validations] Display missing values")
    click.echo("\nMissing values (train):")
    click.echo(train_df.isna().mean())
    click.echo("\nMissing values (test):")
    click.echo(test_df.isna().mean())

    # Save outputs
    click.echo("[Preprocessing] Finished process, saving datasets...")
    try:
        train_dir = os.path.dirname(output_train)
        test_dir = os.path.dirname(output_test)

        # To ensure the output paths exist
        if train_dir:
            os.makedirs(train_dir, exist_ok=True)
        if test_dir:
            os.makedirs(test_dir, exist_ok=True)

        train_df.to_csv(output_train, index=False)
        click.echo(f"Successfully saved train dataset to: {output_train}")
        test_df.to_csv(output_test, index=False)
        click.echo(f"Successfully saved test dataset to: {output_test}")
    except Exception as e:
        click.echo("Error saving output files:", e)
        sys.exit(1)


if __name__ == "__main__":
    clean_data()
