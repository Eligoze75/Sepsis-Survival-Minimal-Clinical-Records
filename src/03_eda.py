import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")


def eda(processed_training_data, plot_to):
    """Generate all EDA plots and save them to PNG files."""
    ###### FIGURE 1: Histogram, Bar Plot, and Heat Map #####
    
    # Load data
    train_df = pd.read_csv(processed_training_data)
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # Histogram of Age grouped by target
    sns.histplot(data=train_df, x="age", hue="hospital_outcome_cat", bins=30, ax=axes[0])
    axes[0].set_title("Histogram of Age grouped by Hospital Outcome")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Outcome", labels=["Died", "Survived"])
    
    # Barplot of Number of Episodes
    train_df["episode_number"].value_counts().sort_index().plot(
        kind="bar", color="#49759c", ax=axes[1]
    )
    axes[1].set_title("Number of Episodes Distribution")
    axes[1].set_xlabel("Episode Number")
    axes[1].set_ylabel("Count")
    
    # Heatmap: Hospital Outcome vs Sex
    pivot_sex_target = train_df.pivot_table(
        index="sex", columns="hospital_outcome_cat", aggfunc="size", fill_value=0
    )
    sns.heatmap(pivot_sex_target, annot=True, fmt=".0f", cmap="Blues", ax=axes[2])
    axes[2].set_title("Count of Cases by Sex and Hospital Outcome")
    axes[2].set_xlabel("Hospital Outcome")
    axes[2].set_ylabel("Sex")
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_to, "histogram_bar_heatmap.png"))
    plt.close(fig)

    ###### FIGURE 2: Boxplots #####
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))
    
    # Boxplot of Age by Hospital Outcome
    sns.boxplot(
        data=train_df,
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
        data=train_df, x="episode_number", y="age", hue="hospital_outcome_cat", ax=axes[1]
    )
    axes[1].set_xlabel("Episode Number")
    axes[1].set_ylabel("Age")
    axes[1].set_title("Boxplot of Age by Hospital Outcome")
    axes[1].legend_.remove()
    # Boxplot of Age by Sex and Hospital Outcome
    sns.boxplot(data=train_df, x="sex", y="age", hue="hospital_outcome_cat")
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
    fig.savefig(os.path.join(plot_to, "boxplots.png"))
    plt.close(fig)


    ###### FIGURE 3: Correlation Heat Maps #####
    df_encoded = train_df[["age", "sex", "episode_number", "hospital_outcome"]].copy()

    # Convert categories to 0/1
    df_encoded["sex"] = df_encoded["sex"].astype("category").cat.codes
    correlation_matrix = df_encoded.corr()
    
    fig = plt.figure(figsize=(5, 4))
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
    fig.savefig(os.path.join(plot_to, "correlation_heatmap.png"))
    plt.close(fig)

if __name__ == '__main__':
    eda()