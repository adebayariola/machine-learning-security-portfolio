# HTTPS Traffic Classification Pipelines

[← Portfolio home](../../../README.md) · [Security projects](../../../README.md#security)

> **Publication status:** Source code included.

## Research question

Can engineered network features distinguish HTTPS traffic classes?

## Dataset

HTTPS-clf-dataset.csv; Kaggle HTTPS Traffic Classification

## Tools and methods

Python, pandas, scikit-learn, PyTorch, joblib and Matplotlib

## What I implemented

- Implemented a Random Forest baseline using the TYPE label and numeric traffic features, with saved model, class distribution, classification report and confusion-matrix outputs.
- Adapted a local deep-learning runner to compare DNN, CNN, RNN, LSTM and GRU architectures using stratified 70/15/15 splits, train-fitted standardisation and class-balanced weights.
- Included a Logistic Regression stacking stage over base-model outputs and evaluation with macro precision, recall, F1 and ROC-AUC. Published upstream paper scores are not presented as personal results.

## Results and limitations

The upstream repository is associated with a published study; contributions here concern adaptation and local pipeline engineering.

These entries describe inspected implementation work. Published reference scores and unverified draft comparisons are not treated as results of this code. Dataset adapters do not establish completed runs on every supported dataset.

## Code and reproduction

[Pipeline script](run_pipeline.py) · [Implementation and command guide](docs/implementation-guide.md)

Run commands from this project directory after installing its dependencies and obtaining the relevant dataset. The implementation guide describes available commands, not evidence that every experiment has been completed.

Dependencies: `python -m pip install -r requirements.txt`. Inspect `python run_pipeline.py --help` for dataset arguments before training.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
