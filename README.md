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

The dataset includes demographic and job-related features:

- Age, Gender, Education, Department, Job Role, Years at Company, etc.
- Target variable: `Attrition` (Yes/No)

Initial preprocessing removed constant or non-informative features such as `EmployeeID`, `EmployeeCount`, `Over18`, and `StandardHours`.

---

## EDA and Preprocessing

- Identified variable distributions, outliers, and class balance.
- No resampling needed (target was nearly balanced).
- Missing values were handled using:
  - Median imputation for numerical features.
  - Mode imputation for ordinal/categorical features.
- Categorical features were encoded using OneHotEncoding or Label Encoding as appropriate.

---

## Modeling Approach

### Baseline Models

- **Logistic Regression** and **XGBoost** were used for benchmarking.
- Random Forest outperformed other models in recall and F1-score.

### Random Forest with Hyperparameter Tuning

- Tuned via `RandomizedSearchCV` with 5-fold cross-validation.
- Best parameters: `n_estimators=200`, `max_features='log2'`, etc.

---

## Feature Selection

- Feature importance was computed from the trained Random Forest.
- Only numerical features with importance > 0.01 were retained.
- 19 key features were selected for the final pipeline.

---

## Final Pipeline

- A complete `Pipeline` was built using scikit-learn:
  - Median imputation for numerical features.
  - Random Forest classifier (already trained and tuned).
- Pipeline was trained on the full dataset.
- Saved to `rf_pipeline_selected_features.pkl`

The standalone model (without preprocessing steps) was also saved as `rf_model_only.pkl`.

---

## Evaluation Metrics

- **Recall**: Prioritized to detect true attrition cases.
- **F1 Score**: Balance between precision and recall.
- **ROC AUC**: 0.994
- **Confusion Matrix** and classification report included in the notebook.

---

## Usage Guide

To use the pipeline for new predictions:

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('rf_pipeline_selected_features.pkl')

# Load new employee data
new_data = pd.read_csv('new_employees.csv')

# Predict
predictions = pipeline.predict(new_data)
probas = pipeline.predict_proba(new_data)[:, 1]

To use only the trained model (with preprocessed data):
model = joblib.load('rf_model_only.pkl')
probas = model.predict_proba(preprocessed_data)[:, 1]

---

## Conclusion

A machine learning model was built to estimate the probability of employee attrition using historical HR data. The final model is based on a Random Forest classifier, trained with optimized hyperparameters and a reduced set of selected numerical features. It was integrated into a complete pipeline with preprocessing and saved in .pkl format. Both the pipeline and the standalone model can be used to generate attrition predictions on new employee data.
