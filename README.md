# Sepsis Survival Minimal Clinical Records

This project focuses on predicting patient survival using machine learning techniques, with the goal of supporting faster and more informed clinical decision making.

## Project Overview

[The dataset](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records) contains 110,204 hospital admissions from 84,811 patients in Norway between 2011 and 2012. All patients were diagnosed with infections, systemic inflammatory response syndrome, sepsis caused by microbes, or septic shock.

The prediction task is to determine whether a patient survived or deceased approximately nine days after their medical record was collected. Sepsis is a life-threatening condition characterized by an extreme immune response to infection, often leading to organ failure or death. Its progression can be extremely rapid, sometimes within one hour, making timely diagnosis and intervention challenging. Many laboratory tests require more time than what clinicians have available when treating severe sepsis cases.

Being able to predict patient survival quickly, using only a small set of easily obtainable medical features, can support faster decision making and potentially improve outcomes.

## Goals

This project includes the following components:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and preparation
3. Training classification models to predict patient survival
4. Model evaluation and explainability
5. Final conclusions and discussion

## Repository Structure

```bash
project/
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Cleaned and prepared data
├── notebooks/             # EDA and experimentation
├── src/                   # Data prep, training, evaluation scripts
├── models/                # Saved model artifacts
├── img/                   # Figures and output reports
├── environment.yml        # Reproducible environment specification
└── README.md
```

## Environment Setup

To reproduce the environment, run:

```bash
conda env create -f environment.yml
conda activate sepsis_survival_venv
```
