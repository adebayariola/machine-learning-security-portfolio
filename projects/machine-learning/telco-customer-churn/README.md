# Telecom Customer Churn Prediction

[← Portfolio home](../../../README.md) · [Machine Learning projects](../../../README.md#machine-learning)

> **Publication status:** Code and saved evaluation included.

## Research question

Which customers are likely to leave a telecom service?

## Dataset

IBM Telco Customer Churn; WA_Fn-UseC_-Telco-Customer-Churn.csv

## Tools and methods

Python, pandas, scikit-learn and reproducible training scripts

## What I implemented

- Prepared 7,043 customer records and 19 predictors; converted billing values, stratified the train/test split and built pipeline-based imputation, scaling and one-hot encoding.
- Compared class-weighted Logistic Regression and Random Forest. Saved held-out ROC-AUC values were 0.8413 and 0.8372; Logistic Regression churn recall was 0.7834. Exported metrics and run metadata.

## Results and limitations

Codex/2026-09-15/n-2/projects/telecom-churn-prediction: data, train_churn_models.py, model_metrics.csv, run_metadata.json and README.

## Held-out Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.7551 | 0.5276 | 0.7406 | 0.6162 | 0.8372 |

## Best ROC-AUC Result
Logistic Regression achieved a held-out ROC-AUC of 0.8413, with recall of 0.7834 and F1 of 0.6136 for customers who churned.

## Reproduce
Run `python src/train_churn_models.py` from this project directory.

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`, then install `requirements.txt`. [Saved metrics](outputs/model_metrics.csv) · [Run metadata](outputs/run_metadata.json).

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
