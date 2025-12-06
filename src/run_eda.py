import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click

TRAIN_FILENAME = "./data/processed/sepsis_train.csv"
UNIVARIATE_FILENAME = "univariate_visualization"
MILTIVARIATE_FILENAME = "multivariate_visualization"
CORR_HEATMAP_FILENAME = "correlation_heatmap"
DEFAULT_EXTENTION = "png"
DEFAULT_SHOW = True
CORR_COLS = ["age", "sex", "episode_number", "hospital_outcome"]


def load_train_df(filename):
    """Load a training dataset from a CSV file.

    This function reads the specified CSV file
    into a pandas DataFrame.

    Args:
        filename (str): Path to the CSV file to load.

    Returns:
        pandas.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    print(f"\n[Loading Data] {filename}...\n")
    return pd.read_csv(filename)


def compute_descriptive_stats(df):
    """Compute and display descriptive statistics for the dataset.

    This function prints summary statistics for numerical columns, category counts
    for categorical features, and the proportion of missing values for each
    column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset to analyze.

    Returns:
        None: This function prints results but does not return any value.
    """
    print("\n[Descriptive statistics] summary:\n\n")
    print(df.describe())
    print("\n[Descriptive statistics] Counts by category:\n\n")
    print("\nNumber of observations of each Sex\n")
    print(df["sex"].value_counts(True, dropna=False))
    print("\nNumber of observations of each Hospital Outcome (target)\n")
    print(df["hospital_outcome_cat"].value_counts(True, dropna=False))
    print("\n[Descriptive statistics] Missing values ratio per column:\n")
    print(df.isna().mean())


def get_univariate_subplots(df, save_filename, extension, show):
    """Generate and save a set of univariate visualizations.

    This function creates a figure with three subplots to summarize key
    univariate and bivariate relationships in the dataset:

    1. A histogram of age, grouped by the target variable.
    2. A bar plot showing the distribution of episode counts.
    3. A heatmap showing the cross-tabulation of sex and hospital outcome.

    The resulting figure is saved to the img/ folder and optionally displayed.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to plot.
        save_filename (str): Base filename (without extension) for saving the figure.
        extension (str): File extension specifying the output image format
            (e.g., "png", "pdf", "svg").
        show (bool): If True, the generated plot is displayed.
            If False, the plot is only saved to disk.

    Returns:
        None: The function saves and optionally displays the figure but
        does not return any value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # Histogram of Age grouped by target
    sns.histplot(data=df, x="age", hue="hospital_outcome_cat", bins=30, ax=axes[0])
    axes[0].set_title("Histogram of Age grouped by Hospital Outcome")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Outcome", labels=["Died", "Survived"])

    # Barplot of Number of Episodes
    df["episode_number"].value_counts().sort_index().plot(
        kind="bar", color="#49759c", ax=axes[1]
    )
    axes[1].set_title("Number of Episodes Distribution")
    axes[1].set_xlabel("Episode Number")
    axes[1].set_ylabel("Count")

    # Heatmap: Hospital Outcome vs Sex
    pivot_sex_target = df.pivot_table(
        index="sex", columns="hospital_outcome_cat", aggfunc="size", fill_value=0
    )
    sns.heatmap(pivot_sex_target, annot=True, fmt=".0f", cmap="Blues", ax=axes[2])
    axes[2].set_title("Count of Cases by Sex and Hospital Outcome")
    axes[2].set_xlabel("Hospital Outcome")
    axes[2].set_ylabel("Sex")

    plt.tight_layout()
    plt.savefig(
        f"img/{save_filename}.{extension}",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    if show:
        plt.show()


def get_multivariate_subplots(df, save_filename, extension, show):
    """Generate and save multivariate visualizations.

    This function creates a figure with three boxplots to examine how the
    distribution of age varies across different categorical variables:

    1. Age by hospital outcome.
    2. Age by episode number, grouped by hospital outcome.
    3. Age by sex, grouped by hospital outcome.

    The figure is saved to disk and optionally displayed.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to visualize.
        save_filename (str): Base filename (without extension) under which the figure
            will be saved.
        extension (str): File extension for the output image (e.g., ``"png"``,
            ``"pdf"``).
        show (bool): If True, displays the generated plots. If False,
            only saves the figure to the ``img/`` directory.

    Returns:
        None: The function saves and optionally displays the figure but does not
        return any value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # Boxplot of Age by Hospital Outcome
    sns.boxplot(
        data=df,
        x="hospital_outcome_cat",
        y="age",
        hue="hospital_outcome_cat",
        ax=axes[0],
    )
    axes[0].set_title("Boxplot of Age by Hospital Outcome")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Age")

    # Boxplot of Age by Hospital Outcome and Episode Number
    sns.boxplot(
        data=df, x="episode_number", y="age", hue="hospital_outcome_cat", ax=axes[1]
    )
    axes[1].set_xlabel("Episode Number")
    axes[1].set_ylabel("Age")
    axes[1].set_title("Boxplot of Age by Hospital Outcome")
    axes[1].legend_.remove()
    # Boxplot of Age by Sex and Hospital Outcome
    sns.boxplot(data=df, x="sex", y="age", hue="hospital_outcome_cat")
    axes[2].set_xlabel("Sex")
    axes[2].set_ylabel("Age")
    axes[2].set_title("Boxplot of Age by Sex and Hospital Outcome")
    axes[2].legend_.remove()

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Outcome",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.075),
    )

    plt.tight_layout()
    plt.savefig(
        f"img/{save_filename}.{extension}",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    if show:
        plt.show()


def get_corr_heatmap(df, use_cols, save_filename, extension, show):
    """Generate and save a correlation heatmap for selected columns.

    This function computes a correlation matrix using only the columns
    specified in ``use_cols``. It then generates a heatmap visualization
    of the correlations, saves the resulting figure, and optionally displays it.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        use_cols (list[str]): List of column names for which the correlation
            matrix will be computed and visualized.
        save_filename (str): Base filename (without extension) under which the
            heatmap image will be saved.
        extension (str): File extension for the saved image
            (e.g., ``"png"``, ``"pdf"``, ``"svg"``).
        show (bool): If True, displays all visualization plots. If False,
            the plots will only be saved to the ``img/`` directory.

    Returns:
        None: The function saves and optionally displays the correlation heatmap,
        but does not return any value.
    """
    # Convert categories to 0/1
    df["sex"] = df["sex"].astype("category").cat.codes
    correlation_matrix = df[use_cols].corr()

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        linewidths=0.5,
        linecolor="black",
    )
    plt.title("Correlation Heatmap of Sepsis Numerical Features")
    plt.tight_layout()
    plt.savefig(
        f"img/{save_filename}.{extension}",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    if show:
        plt.show()


@click.command()
@click.option(
    "--filename",
    default=TRAIN_FILENAME,
    show_default=True,
    type=str,
    help="Path to the input file.",
)
@click.option(
    "--file_extention",
    default=DEFAULT_EXTENTION,
    show_default=True,
    type=str,
    help="File format/extension of the output image (e.g., png, jpg, pdf).",
)
@click.option(
    "--use_corr_cols",
    default=CORR_COLS,
    show_default=True,
    type=list,
    help="List of column names to include in the correlation analysis.",
)
@click.option(
    "--show_visualizations",
    default=DEFAULT_SHOW,
    show_default=True,
    type=bool,
    help="Show the generated plots. If false, plots are saved but not displayed.",
)
def main(filename, file_extention, use_corr_cols, show_visualizations):
    """Runs the EDA steps.

    This function runs all steps of the EDA:

    1. Loads data.
    2. Computes Univariate and Bivariate visualizations.
    3. Computes Multivariate visualizations.
    4. Computes correlation matrix.

    Args:
        filename (str): The input DataFrame containing the data.
        file_extention (str): File extension for the saved image
            (e.g., ``"png"``, ``"pdf"``, ``"svg"``).
        use_corr_cols (str): Base filename (without extension) under which the
            heatmap image will be saved.
        extension (str): File extension for the saved image
            (e.g., ``"png"``, ``"pdf"``, ``"svg"``).
        show (bool): If True, displays the plot. If False,
            the plot is only saved to the ``img/`` directory.

    Returns:
        None: The function saves and optionally displays the generated visualizations,
        but does not return any value.
    """
    print(" " * 35, "EXPLORATORY DATA ANALYSIS\n\n")
    df = load_train_df(filename)
    compute_descriptive_stats(df)
    print("\n[Univariate and Bivariate visualizations]\n")
    get_univariate_subplots(
        df, UNIVARIATE_FILENAME, extension=file_extention, show=show_visualizations
    )
    print("\n[Univariate and Bivariate visualizations]\n")
    get_multivariate_subplots(
        df, MILTIVARIATE_FILENAME, extension=file_extention, show=show_visualizations
    )
    print("\n[Correlation Heatmap]\n\n")
    get_corr_heatmap(
        df,
        use_corr_cols,
        CORR_HEATMAP_FILENAME,
        extension=file_extention,
        show=show_visualizations,
    )


if __name__ == "__main__":
    main()
