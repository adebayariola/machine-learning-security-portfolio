# Residential Property Price Prediction

[← Portfolio home](../../../README.md) · [Machine Learning projects](../../../README.md#machine-learning)

> **Publication status:** Project summary; source not published.

## Research question

How do XGBoost and TabNet compare for residential property valuation?

## Dataset

King County Sales by Andy Krause, Kaggle; kingCoData

## Tools and methods

Python, pandas, XGBoost, PyTorch-based TabNet, Matplotlib, Seaborn and Google Colab

## What I implemented

- Compared XGBoost and TabNet for Seattle and the rest of King County using structural-only and full feature sets from housing records covering 2015–2025.
- Engineered property age, renovation age, garage area, scenic-view indicators and Haversine distances; cleaned records, encoded categories, treated outliers and applied log transformations.
- Used five-fold cross-validation, early stopping, MAE, RMSE, MAPE and R². The dissertation reports full-feature XGBoost R² of 0.92 for Seattle and 0.96 for King County.

## Results and limitations

Downloads/Submissions/ITNPBD5/3334661.pdf; final dissertation results are distinct from subsequent local experiments.

This publication currently contains a project description only. Source files and executable results are not included here.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
