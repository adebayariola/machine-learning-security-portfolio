# HTTPS Traffic Classification Pipelines

## Data
HTTPS-clf-dataset.csv; Kaggle HTTPS Traffic Classification

## Tools
Python, pandas, scikit-learn, PyTorch, joblib and Matplotlib

## Work completed
- Implemented a Random Forest baseline using the TYPE label and numeric traffic features, with saved model, class distribution, classification report and confusion-matrix outputs.
- Adapted a local deep-learning runner to compare DNN, CNN, RNN, LSTM and GRU architectures using stratified 70/15/15 splits, train-fitted standardisation and class-balanced weights.
- Included a Logistic Regression stacking stage over base-model outputs and evaluation with macro precision, recall, F1 and ROC-AUC. Published upstream paper scores are not presented as personal results.

## Evidence and limitations
The upstream repository is associated with a published study; contributions here concern adaptation and local pipeline engineering.

Raw datasets and private operational records are not distributed.

## Source files
- [run_pipeline.py](https-traffic-classification-pipelines__run_pipeline.py)
