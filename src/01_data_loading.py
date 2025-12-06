import click
import os
from utils import *
from validations import *
import sys

@click.command()
@click.option(
    "--filename",
    "-n",
    required=True,
    help="Name of CSV file INSIDE the UCI ZIP archive to load",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Path to directory where raw data will be written to",
)

def download_data(filename, output):
    """
    Downloads the UCI Sepsis Survival dataset ZIP,
    extracts the specified CSV file,
    validates its format,
    and saves the CSV locally.
    """

    click.echo(f"Attempting to load '{filename}' from UCI dataset...")

    # Load the data using your existing function
    try:
        df = load_ucisepsis(filename)
        click.echo("File successfully downloaded and extracted.")
    except Exception as e:
        click.echo(f"ERROR: Could not load file '{filename}'")
        click.echo(str(e))
        sys.exit("Stopping execution due to invalid file format.")

    # Validate format using your existing checker
    try:
        check_file_format(filename)
        click.echo("File format validated successfully.")
    except AssertionError as e:
        click.echo("File format validation failed.")
        click.echo(str(e))
        sys.exit("Stopping execution due to invalid file format.")
    
    # Save output
    try:
        output_dir = os.path.dirname(output)
        # To ensure the output path exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
        df.to_csv(output, index=False)
        click.echo(f"Saved dataset to: {output}")
    except Exception as e:
        click.echo(f"Failed to save file to {output}")
        click.echo(str(e))
        sys.exit(1)


if __name__ == "__main__":
    download_data()