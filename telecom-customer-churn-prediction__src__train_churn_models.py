"""Train and compare reproducible telecom churn classifiers.

The test set is held out before imputation, encoding, scaling or model fitting to
avoid leaking information from evaluation data into training.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS = ROOT / "outputs"
RANDOM_STATE = 42


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.drop(columns=["customerID"])
    target = data.pop("Churn").map({"Yes": 1, "No": 0})
    return data, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def score(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "f1": round(f1_score(y_test, predicted), 4),
        "roc_auc": round(roc_auc_score(y_test, probability), 4),
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    data = pd.read_csv(DATA)
    x, y = clean_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    metrics = []
    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(x_train)), ("classifier", classifier)]
        )
        pipeline.fit(x_train, y_train)
        metrics.append({"model": name, **score(pipeline, x_test, y_test)})

    results = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    results.to_csv(OUTPUTS / "model_metrics.csv", index=False)

    metadata = {
        "dataset": "IBM Telco Customer Churn",
        "rows": int(len(data)),
        "features_before_target": int(x.shape[1]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "churn_rate": round(float(y.mean()), 4),
        "random_state": RANDOM_STATE,
    }
    (OUTPUTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    best = results.iloc[0]
    table_header = "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    table_rows = "\n".join(
        f"| {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.roc_auc:.4f} |"
        for row in results.itertuples(index=False)
    )
    report = f"""# Telecom Customer Churn Prediction

## Question
Which customer and service characteristics help predict whether a telecom customer will churn?

## Method
- Dataset: IBM Telco Customer Churn ({metadata['rows']:,} customers; {metadata['features_before_target']} predictors).
- Converted `TotalCharges` to numeric and treated invalid blank values as missing.
- Split the data into stratified training and held-out test sets before preprocessing.
- Imputed missing values, scaled numeric columns and one-hot encoded categorical columns inside each model pipeline.
- Compared class-weighted Logistic Regression and Random Forest models.
- Evaluated accuracy, precision, recall, F1 and ROC-AUC; ROC-AUC and recall are useful alongside accuracy because churn is the minority class.

## Held-out Test Results

{table_header}
{table_rows}

## Best ROC-AUC Result
{best['model']} achieved a held-out ROC-AUC of {best['roc_auc']:.4f}, with recall of {best['recall']:.4f} and F1 of {best['f1']:.4f} for customers who churned.

## Reproduce
Run `python src/train_churn_models.py` from this project directory.
"""
    (ROOT / "README.md").write_text(report)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
