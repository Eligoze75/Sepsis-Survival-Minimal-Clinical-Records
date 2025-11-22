# Sepsis Survival Minimal Clinical Records

This project focuses on predicting patient survival using machine learning techniques, with the goal of supporting faster and more informed clinical decision making.

## Project Overview

Sepsis is a life-threatening condition where the body's immune system overreacts to infection, potentially causing organ failure. Quick diagnosis is crucial since sepsis can progress rapidly, sometimes within an hour.

This project explores whether basic patient information Age, Sex, and number of prior Sepsis episodes can predict survival outcomes. We analyzed [a dataset](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records) of over 110,000 hospital admissions from Norway (2011-2012) using a Logistic Regression model.

**Key Findings:**
- Age emerged as the strongest predictor: older patients have lower survival probability (SHAP value: 0.86)
- The model achieved a ROC AUC of 0.59, showing limited predictive power
- Sex and Episode Number had minimal impact on predictions
- The dataset is highly imbalanced: 93% survivors vs. 7% non survivors

**Conclusion:** While Age is an important mortality risk indicator, these basic demographic factors alone are not sufficient for accurate predictions. Additional clinical features (vital signs, lab values, pre existing conditions) would be needed to improve the model's performance.

## Project Components

This project includes the following components:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and preparation
3. Training classification models to predict patient survival
4. Model evaluation and explainability
5. Final conclusions and discussion

## Repository Structure

```bash
Sepsis-Survival-Minimal-Clinical-Records/
├── data/
│   ├── processed/
│   ├── s41598-020-73558-3_sepsis_survival_primary_cohort.csv
│   ├── s41598-020-73558-3_sepsis_survival_study_cohort.csv
│   └── s41598-020-73558-3_sepsis_survival_validation_cohort.csv
├── src/
│   ├── sepsis-predictor-report.ipynb                   # Main analysis notebook
│   └── utils.py                                        # Helper functions
├── models/
├── environment.yml                                     # Conda environment
├── conda-lock.yml                                      # Locked dependencies
├── CONTRIBUTING.md                                     # Contribution guidelines
├── LICENSE
└── README.md
```

## Environment Setup

To reproduce the environment, run:

```bash
conda env create -f environment.yml
conda activate sepsis_survival_venv
```

### Using the Conda Lock File

This repository also includes a conda-lock.yml file generated to ensure full reproducibility across platforms and installations. This file contains exact, fully resolved dependency versions.

To create the environment using the lock file:

Install conda-lock if you do not already have it:

```bash
pip install conda-lock
```

or via conda:

```bash
conda install -c conda-forge conda-lock
```

Generate the environment from the lock file:

```bash
conda-lock install --name sepsis_survival_venv conda-lock.yml
```

This will create (or update) the environment with the exact versions pinned in the lock file.

Activate the environment:

```bash
conda activate sepsis_survival_venv
```

## Contributing

We welcome contributions! Whether you're a data scientist, clinician, or machine learning enthusiast, your input can help improve this project. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Acknowledgments

Thank you for exploring this project! We hope this analysis provides valuable insights into sepsis prediction and inspires further research. Whether you're here to learn, contribute, or simply explore, we appreciate your interest.

Happy learning! 🎓

---

*For questions or discussions, feel free to open an issue or reach out to the maintainers.*
