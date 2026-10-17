# HTTPS Traffic Classification

This folder contains a full training and evaluation pipeline for:

`/path/to/local/Downloads/HTTPS-clf-dataset.csv`

The pipeline uses `TYPE` as the target label and trains a Random Forest classifier
on the numeric engineered features in the dataset.

## Run

```bash
python3 https_traffic_classification/run_pipeline.py
```

Quick smoke test:

```bash
python3 https_traffic_classification/run_pipeline.py --sample 10000 --trees 50
```

Outputs are written to:

`https_traffic_classification/outputs/`

Generated files include:

- `https_traffic_classifier.joblib`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `label_distribution.csv`
- `label_distribution.png`
