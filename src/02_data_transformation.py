import click
import pandas as pd
import os

from clean_data import clean_survival_data


@click.command()
@click.option("--input-train", required=True, help="Path to raw TRAIN CSV")
@click.option("--input-test", required=True, help="Path to raw TEST CSV")
@click.option("--output-train", required=True, help="Path to cleaned TRAIN CSV")
@click.option("--output-test", required=True, help="Path to cleaned TEST CSV")
def main(input_train, input_test, output_train, output_test):
    """CLI entry point for cleaning sepsis survival datasets."""

    train_df = pd.read_csv(input_train)
    test_df = pd.read_csv(input_test)

    train_clean = clean_survival_data(train_df)
    test_clean = clean_survival_data(test_df)

    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)

    train_clean.to_csv(output_train, index=False)
    test_clean.to_csv(output_test, index=False)

    click.echo("Cleaned datasets saved successfully.")


if __name__ == "__main__":
    main()
