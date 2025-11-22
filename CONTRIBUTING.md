# Feedback and Contribution

We welcome any input, feedback, bug reports, and contributions to the **Sepsis Survival Prediction Project**. This project aims to explore whether basic clinical and demographic factors, specifically Age, Sex, and Sepsis Episode Number can be used to predict sepsis survival outcomes using machine learning techniques.

Your contributions can help us:
- Improve model performance and predictive accuracy
- Incorporate additional clinical features
- Enhance data analysis methodologies
- Refine visualizations and interpretability methods
- Expand documentation and reproducibility

All contributions, suggestions, and feedback you submit are accepted under the [Project's license](LICENSE). You represent that if you do not own copyright in the code that you have the authority to submit it under the [Project's license](LICENSE). All feedback, suggestions, or contributions are not confidential.

## Why Your Contribution Matters

Our initial analysis revealed that while Age is a significant predictor of sepsis survival (with a mean SHAP value of 0.86), the model achieved a modest ROC AUC of 0.59, indicating limited discriminative ability. The analysis highlights several opportunities for improvement:

- **Feature Engineering**: Incorporating additional clinical variables such as vital signs (temperature, blood pressure, heart rate), laboratory values (white blood cell count, lactate levels), and pre existing conditions.
- **Model Architecture**: Exploring more sophisticated algorithms beyond logistic regression.
- **Domain Expertise**: Clinical insights from healthcare professionals to guide feature selection.

We believe that collaborative efforts from data scientists, clinicians, and machine learning practitioners can significantly enhance the predictive power and clinical utility of this work.

## How You Can Contribute

We welcome contributions in various forms:

### 1. Bug Reports and Issues
- Report errors, inconsistencies, or issues in the code or analysis
- Identify potential biases or methodological concerns

### 2. Feature Suggestions
- Propose new clinical features to include in the model
- Recommend alternative machine learning algorithms or techniques
- Suggests better and richier datasets!

### 3. Code Contributions
- Implement feature engineering pipelines
- Add new model architectures or ensemble methods
- Optimize preprocessing and data handling

### 4. Data Science and Clinical Expertise
- Provide domain knowledge about sepsis and clinical variables
- Validate the clinical relevance of findings

## How To Contribute

### Setting Up Your Environment

1. **Fork the Repository**: Create a personal copy of the repository by forking it to your GitHub account.

2. **Clone the Repository**: Clone your forked repository to your local machine:

```bash
git clone https://github.com/YOUR-USERNAME/Sepsis-Survival-Minimal-Clinical-Records.git
cd Sepsis-Survival-Minimal-Clinical-Records
```

3. **Set Up the Environment**: This project uses conda for environment management. We provide both `environment.yml` and `conda-lock.yml` files. The `conda-lock.yml` file ensures reproducibility across different operating systems (Linux, macOS, Windows).

**Option A: Using conda-lock (Recommended for full reproducibility)**

If you don't have conda-lock installed, first install it:

```bash
conda install -c conda-forge conda-lock
```

Then install the environment:

```bash
conda-lock install -n sepsis_survival_venv conda-lock.yml
conda activate sepsis_survival_venv
```

**Option B: Using environment.yml (Faster setup)**

```bash
conda env create -f environment.yml
conda activate sepsis_survival_venv
```

The environment includes all necessary dependencies:
- Python 3.12
- pandas, numpy, scipy
- scikit-learn for machine learning
- matplotlib, seaborn, plotly for visualization
- shap for model interpretability
- jupyterlab for notebook development
- black and flake8 for code formatting and linting


### Creating a Branch

Once your local environment is set up, create a new git branch for your contribution (always create a new branch instead of making changes to the main branch):

```bash
git checkout -b your_branch_name
```

Try to use a branch name that mentions your user name and that is related to your contribution.

### Making Changes

With your branch checked out, make the desired changes to the codebase. Here are some guidelines:

- **Code Style**: Follow PEP 8 guidelines for Python code
- **Documentation**: Add docstrings to functions and classes
- **Comments**: Include clear comments explaining complex logic
- **Reproducibility**: Ensure your code is reproducible by setting random seeds where applicable
- **Modularity**: Break down complex functions into smaller, reusable components (these must be in the ```src/utils.py```)

### Testing Your Changes

Before submitting your changes, ensure that:

1. **Your code runs without errors**: Test all modified code paths
2. **Results are reproducible**: Run your analysis multiple times to verify consistency
3. **Notebooks execute completely**: If modifying notebooks, restart the kernel and run all cells
4. **Dependencies are documented**: Update `requirements.txt` if you add new packages

### Creating a Pull Request

Provide a clear title and detailed description of your changes:
- What problem does it solve?
- What are the key results or improvements?
- Are there any limitations or concerns?

Your PR will be reviewed, and you may receive feedback or requests for changes.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Be respectful and considerate in communications
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully


## Questions and Discussions

If you have questions or want to discuss ideas before contributing:

- Open an issue with the `question` or `discussion` label
- Reach out to the maintainers
- Share your ideas and get feedback before investing significant time

## Acknowledgments

We appreciate your interest in improving sepsis survival prediction. Every contribution, no matter how small, helps advance our understanding and can potentially impact clinical decision making in the future.

Thank you for considering contributing to this project! Together, we can work towards better predictive tools for sepsis survival outcomes.

---

*This document was inspired by the [Altair CONTRIBUTING.md](https://github.com/vega/altair/blob/main/CONTRIBUTING.md) and adapted for the Sepsis Survival Prediction Project.*

