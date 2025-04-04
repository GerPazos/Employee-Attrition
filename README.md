# 💼 Employee Attrition Prediction

This project aims to predict employee attrition based on various personal and work-related features.

## Table of Contents
- [Project Description](#project-description)
- [Key Concepts](#key-concepts)
- [Dataset Overview](#dataset-overview)
- [EDA and Preprocessing](#eda-and-preprocessing)
- [Modeling Approach](#modeling-approach)
  - [Baseline Models](#baseline-models)
  - [Random Forest with Hyperparameter Tuning](#random-forest-with-hyperparameter-tuning)
- [Feature Selection](#feature-selection)
- [Final Pipeline](#final-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage Guide](#usage-guide)
- [Conclusion and Run Instructions](#conclusion-and-run-instructions)

---

## Project Description

The goal of this project is to predict whether an employee will leave the company (attrition) based on various personal and job-related features. By analyzing historical HR data, the project builds a machine learning model to assess attrition risk.

This work was done as part of a data science challenge and includes both exploratory data analysis and the development of a production-ready machine learning pipeline.

---

## Key Concepts

- **Binary Classification**: Predict whether an employee will leave (`Yes`) or stay (`No`).
- **Imputation**: Handling missing values with statistical methods.
- **Feature Engineering**: Selecting and encoding the most relevant features.
- **Hyperparameter Tuning**: Using `RandomizedSearchCV` to optimize model performance.
- **Model Evaluation**: Focusing on recall and F1-score due to the business goal.

---

## Dataset Overview

The dataset includes demographic and job-related features such as:

- Age, Gender, Education, Department, Job Role, Years at Company, etc.
- Target variable: `Attrition` (Yes/No)

Initial preprocessing removed constant or non-informative features such as `EmployeeID`, `EmployeeCount`, `Over18`, and `StandardHours`.

---

## EDA and Preprocessing

- Variable distributions, outliers, and class balance were explored.
- No resampling was needed (target was nearly balanced).
- Missing values were handled using:
  - Median imputation for skewed numerical features with outliers.
  - Mode imputation for ordinal features.
- Categorical features were encoded using Label Encoding (for binary) and One-Hot Encoding (for multi-category).

---

## Modeling Approach

### Baseline Models

- **Logistic Regression** and **XGBoost** were trained for benchmarking.
- Random Forest outperformed both in recall and F1-score.

### Random Forest with Hyperparameter Tuning

- Hyperparameters were optimized using `RandomizedSearchCV` with 5-fold cross-validation.
- Best configuration included `n_estimators=200`, `max_features='log2'`, `min_samples_split=5`, etc.

---

## Feature Selection

- Feature importances were extracted from the trained Random Forest model.
- Only numerical features with importance > 0.01 were retained.
- A total of 19 features were selected for the final model.

---

## Final Pipeline

- A complete scikit-learn `Pipeline` was created with:
  - Median imputation for selected numerical features.
  - Tuned Random Forest classifier.
- The pipeline was trained on the full dataset and saved as:

```
rf_pipeline_selected_features.pkl
```

- The standalone trained model (without preprocessing) was also saved as:

```
rf_model_only.pkl
```

---

## Evaluation Metrics

- **Recall**: prioritized to detect employees at risk of leaving.
- **F1 Score**: chosen to balance recall and precision.
- **ROC AUC**: 0.994, indicating excellent class separability.
- Additional outputs: classification report, confusion matrix, and ROC curve.

---

## Usage Guide

To use the pipeline for making predictions:

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('rf_pipeline_selected_features.pkl')

# Load new employee data (in this case, the same data used for training is reused for demonstration)
new_employees = pd.read_pickle('attrition_dataset.pkl')

# Predict
predictions = pipeline.predict(new_employees)
probas = pipeline.predict_proba(new_employees)[:, 1]

# Save results
results = new_employees.copy()
results['Attrition_Prediction'] = predictions
results['Attrition_Probability'] = probas
results.to_csv('attrition_predictions.csv', index=False)
```

To use the trained model only (with data already preprocessed):

```python
model = joblib.load('rf_model_only.pkl')
probas = model.predict_proba(preprocessed_data)[:, 1]
```

---

## Conclusion and Run Instructions

A machine learning model was developed to estimate the probability of employee attrition using historical HR data. The final solution is based on a Random Forest classifier trained with optimized hyperparameters and a selected set of numerical features. The model was embedded into a pipeline and saved in `.pkl` format for future use.

To reproduce the results:
- Clone this repository
- Run the notebook from start to finish (`.ipynb`)
- Replace the demo employee dataset with your own for prediction
