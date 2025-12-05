import click
import pandas as pd
import sys


@click.command()
@click.option("--input-train", "-i", required=True, help="Path to raw TRAIN CSV")
@click.option("--input-test", "-j", required=True, help="Path to raw TEST CSV")
@click.option("--output-train", "-o1", required=True, help="Path to cleaned TRAIN CSV")
@click.option("--output-test", "-o2", required=True, help="Path to cleaned TEST CSV")
def clean_data(input_train, input_test, output_train, output_test):
    """Clean and transform train and test datasets."""

    # Load input files
    try:
        train_df = pd.read_csv(input_train)
        test_df = pd.read_csv(input_test)
    except Exception as e:
        print("Error reading input files:", e)
        sys.exit(1)

    # Rename columns
    rename_map = {
        "age_years": "age",
        "sex_0male_1female": "sex",
        "hospital_outcome_1alive_0dead": "hospital_outcome",
    }
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    # Map sex and outcome
    train_df["sex"] = train_df["sex"].map({0: "male", 1: "female"})
    test_df["sex"] = test_df["sex"].map({0: "male", 1: "female"})

    outcome_map = {0: "Died", 1: "Survived"}
    train_df["hospital_outcome_cat"] = train_df["hospital_outcome"].map(outcome_map)
    test_df["hospital_outcome_cat"] = test_df["hospital_outcome"].map(outcome_map)

    # Report missing values
    print("Missing values (train):")
    print(train_df.isna().mean())
    print("\nMissing values (test):")
    print(test_df.isna().mean())

    # Save outputs
    try:
        train_df.to_csv(output_train, index=False)
        test_df.to_csv(output_test, index=False)
    except Exception as e:
        print("Error saving output files:", e)
        sys.exit(1)

    print(f"Saved cleaned train data to: {output_train}")
    print(f"Saved cleaned test data to: {output_test}")


if __name__ == "__main__":
    clean_data()