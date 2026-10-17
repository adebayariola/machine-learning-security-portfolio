# Telecom Customer Churn Prediction

## Data
IBM Telco Customer Churn; WA_Fn-UseC_-Telco-Customer-Churn.csv

## Tools
Python, pandas, scikit-learn and reproducible training scripts

## Work completed
- Prepared 7,043 customer records and 19 predictors; converted billing values, stratified the train/test split and built pipeline-based imputation, scaling and one-hot encoding.
- Compared class-weighted Logistic Regression and Random Forest. Saved held-out ROC-AUC values were 0.8413 and 0.8372; Logistic Regression churn recall was 0.7834. Exported metrics and run metadata.

## Evidence and limitations
Codex/2026-09-15/n-2/projects/telecom-churn-prediction: data, train_churn_models.py, model_metrics.csv, run_metadata.json and README.

Raw datasets and private operational records are not distributed.

## Source files
- [requirements.txt](telecom-customer-churn-prediction__requirements.txt)
- [run_metadata.json](telecom-customer-churn-prediction__outputs__run_metadata.json)
- [model_metrics.csv](telecom-customer-churn-prediction__outputs__model_metrics.csv)
- [train_churn_models.py](telecom-customer-churn-prediction__src__train_churn_models.py)
