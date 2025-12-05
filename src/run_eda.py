import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_FILENAME = "./data/processed/sepsis_train.csv"
UNIVARIATE_FILENAME = "univariate_visualization"
MILTIVARIATE_FILENAME = "multivariate_visualization"
CORR_HEATMAP_FILENAME = "correlation_heatmap"
CORR_COLS = ["age", "sex", "episode_number", "hospital_outcome"]


def load_train_df(filename):
    print(f"\n[Loading Data] {filename}...\n")
    return pd.read_csv(filename)


def compute_descriptive_stats(df):
    print("\n[Descriptive statistics] summary:\n\n")
    df.describe()
    print("\n[Descriptive statistics] Counts by category:\n\n")
    print("\nNumber of observations of each Sex\n")
    df["sex"].value_counts(True, dropna=False)
    print("\nNumber of observations of each Hospital Outcome (target)\n")
    df["hospital_outcome_cat"].value_counts(True, dropna=False)
    print("\n[Descriptive statistics] Missing values ratio per column:\n")
    df.isna().mean()


def get_univariate_subplots(df, save_filename, extension="png"):
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
    plt.show()


def get_multivariate_subplots(df, save_filename, extension="png"):
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
    plt.show()


def get_corr_heatmap(df, use_cols, save_filename, extension="png"):
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
    plt.savefig(
        f"img/{save_filename}.{extension}",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    print(" " * 35, "EXPLORATORY DATA ANALYSIS\n\n")
    df = load_train_df(TRAIN_FILENAME)
    compute_descriptive_stats(df)
    print("\n[Univariate and Bivariate visualizations]\n")
    get_univariate_subplots(df, UNIVARIATE_FILENAME, extension="png")
    print("\n[Univariate and Bivariate visualizations]\n")
    get_multivariate_subplots(df, MILTIVARIATE_FILENAME, extension="png")
    print("\n[Correlation Heatmap]\n\n")
    get_corr_heatmap(df, CORR_COLS, CORR_HEATMAP_FILENAME, extension="png")
