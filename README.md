# Sepsis Survival Minimal Clinical Records

This project focuses on predicting patient survival using machine learning techniques, with the goal of supporting faster and more informed clinical decision making.

## Project Overview

Sepsis is a life-threatening condition where the body's immune system overreacts to infection, potentially causing organ failure. Quick diagnosis is crucial since sepsis can progress rapidly, sometimes within an hour.

This project explores whether basic patient information Age, Sex, and number of prior Sepsis episodes can predict survival outcomes. We analyzed [a dataset](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records) of over 110,000 hospital admissions from Norway (2011-2012) using a Logistic Regression model.

**Key Findings:** - Age emerged as the strongest predictor: older patients have lower survival probability (SHAP value: 0.86) - The model achieved a ROC AUC of 0.59, showing limited predictive power - Sex and Episode Number had minimal impact on predictions - The dataset is highly imbalanced: 93% survivors vs. 7% non survivors

**Conclusion:** While Age is an important mortality risk indicator, these basic demographic factors alone are not sufficient for accurate predictions. Additional clinical features (vital signs, lab values, pre existing conditions) would be needed to improve the model's performance.

## Project Components

This project includes the following components:

1.  Exploratory Data Analysis (EDA)
2.  Data preprocessing and preparation
3.  Training classification models to predict patient survival
4.  Model evaluation and explainability
5.  Final conclusions and discussion

## Repository Structure

``` bash
Sepsis-Survival-Minimal-Clinical-Records/
├── data
│   ├── processed
│   │   ├── sepsis_test.csv
│   │   ├── sepsis_train.csv
│   │   ├── test_clean.csv
│   │   └── train_clean.csv
│   └── raw
│       ├── s41598-020-73558-3_sepsis_survival_primary_cohort.csv
│       └── s41598-020-73558-3_sepsis_survival_study_cohort.csv
├── reports
│   ├── references.bib
│   ├── sepsis-predictor-report.html
│   ├── sepsis-predictor-report.pdf
│   └── sepsis-predictor-report.qmd
├── results
│   ├── figures
│   │   ├── correlation_heatmap.png
│   │   ├── multivariate_visualization.png
│   │   ├── score_by_target_class.png
│   │   ├── shap_values_plot.png
│   │   └── univariate_visualization.png
│   ├── models
│   │   ├── logistic_reg.pkl
│   │   ├── lr.pkl
│   │   └── random_forest.pkl
│   └── tables
│       ├── classification_metrics.csv
│       ├── missing_vals_ratio.csv
│       ├── model_coefficients.csv
│       ├── sex_valcounts.csv
│       ├── target_valcounts.csv
│       └── train_summary.csv
└── src
│   ├── 01_data_loading.py
│   ├── 02_data_transformation.py
│   ├── 03_run_eda.py
│   ├── 04_modeling_and_evaluation.py
│   ├── sepsis-predictor-report.ipynb.                  # Main analysis notebook
│   ├── utils.py                                        
│   └── validations.py
│
├── docker-compose.yml                                             
├── environment.yml                                     # Conda environment
├── conda-linux-64.lock                                 # Locked dependencies
├── CONTRIBUTING.md                                     # Contribution guidelines
├── CODE_OF_CONDUCT.md
├── Dockerfile
├── LICENSE               
└── README.md
                                  

```

## Dependencies

-   [Docker](https://www.docker.com/)
-   [VS Code](https://code.visualstudio.com/download)
-   [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage

### Setup

> If you are using Mac or Windows, ensure Docker Desktop is running.

1.  Clone this GitHub repository.

### Running the analysis

1.  Using the command line on your computer to go to the project's root directory, then run the following command:

``` bash
docker compose up
```

2.  In the terminal output, find a URL which begins with `http://127.0.0.1:8888/lab?token=` Copy that URL and open it in your web browser.

3.  To run the analysis, open a terminal and run the following commands:

    1.  Load raw datasets - Run twice: once for train, once for test.

        ``` bash
        python src/01_data_loading.py 
        python src/01_data_loading.py --filename s41598-020-73558-3_sepsis_survival_study_cohort.csv
        ```

    2.  Transform datasets - Processes and saves cleaned versions.

        ``` bash
        python src/02_data_transformation.py
        ```

    3.  Run EDA - Generates plots and descriptive stats.

        ``` bash
        python src/03_run_eda.py
        ```

    4.  Train and evaluate the model - Fits pipeline, computes metrics and SHAP, saves model.

        ``` bash
        python src/04_modeling_and_evaluation.py
        ```

    5.  Create analysis report - Generate HTML and PDF reports

        ``` bash
        quarto render reports/sepsis-predictor-report.qmd --to html
        quarto render reports/sepsis-predictor-report.qmd --to pdf
        ```

### Clean up

1.  To stop the container and remove associated resources, press `Ctrl` + `C` in the terminal where the container is running, then enter `docker compose rm`

## **Developer notes**

-   Docker ensures full reproducibility without needing to manually configure dependencies, so anyone can run the environment consistently across operating systems without dependency conflicts.

### **Developer dependencies**

-   `conda` (version 25.7.0 or higher)

-   `conda-lock` (version 3.0.4 or higher)

### **Adding a new dependency**

1.  Create a new branch and add the dependency to the `environment.yml` file.

2.  Run the following command to update the `conda-linux-64.lock` file:

    ``` bash
    `conda-lock -k explicit --file environment.yml -p linux-64`
    ```

3.  Build the Docker image locally to verify it builds successfully and runs as expected.

4.  Commit and push the updates to GitHub. A new Docker image tagged with the commit's SHA will automatically be built and pushed to Docker Hub.

5.  Update the `docker-compose.yml` file and ensure the tag of the new container image is generated in your branch.

6.  Submit a pull request to have these updates merged into the `main` branch.

## Contributing

We welcome contributions! Whether you're a data scientist, clinician, or machine learning enthusiast, your input can help improve this project. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Acknowledgments

Thank you for exploring this project! We hope this analysis provides valuable insights into sepsis prediction and inspires further research. Whether you're here to learn, contribute, or simply explore, we appreciate your interest.

Happy learning! 🎓

------------------------------------------------------------------------

*For questions or discussions, feel free to open an issue or reach out to the maintainers.*
