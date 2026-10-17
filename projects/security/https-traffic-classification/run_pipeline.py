from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATASET = Path("/path/to/local/Downloads/HTTPS-clf-dataset.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET_COLUMN = "TYPE"
DROP_COLUMNS = {"Unnamed: 0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an HTTPS traffic classification model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row count for a quicker smoke test before a full run.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def stratified_sample(df: pd.DataFrame, target: str, sample: int | None, seed: int) -> pd.DataFrame:
    if not sample or sample >= len(df):
        return df

    parts = []
    for _, group in df.groupby(target, group_keys=False):
        group_sample_size = max(1, round(sample * len(group) / len(df)))
        parts.append(group.sample(n=min(group_sample_size, len(group)), random_state=seed))
    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled.head(sample)


def select_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    numeric_columns = df.select_dtypes(include="number").columns
    feature_columns = [c for c in numeric_columns if c not in DROP_COLUMNS and c != target]
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    X = df[feature_columns]
    y = df[target].astype(str)
    return X, y


def save_label_distribution(y: pd.Series, output_dir: Path) -> None:
    counts = y.value_counts().sort_index()
    counts.to_csv(output_dir / "label_distribution.csv", header=["count"])

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="#297373")
    plt.title("HTTPS Traffic Label Distribution")
    plt.xlabel("TYPE")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=160)
    plt.close()


def save_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
    )
    ax.set_title("HTTPS Traffic Classification Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    df = stratified_sample(df, args.target, args.sample, args.seed)
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

    X, y = select_features(df, args.target)
    print(f"Using {X.shape[1]} numeric features; target='{args.target}'")
    print("Label distribution:")
    print(y.value_counts().sort_index().to_string())
    save_label_distribution(y, args.output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.trees,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)
    print(f"Accuracy: {accuracy:.4f}")
    print(report_text)

    save_confusion_matrix(model, X_test, y_test, args.output_dir)

    metrics = {
        "dataset": str(args.dataset),
        "rows": int(len(df)),
        "features": list(X.columns),
        "target": args.target,
        "test_size": args.test_size,
        "trees": args.trees,
        "seed": args.seed,
        "accuracy": accuracy,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    joblib.dump(model, args.output_dir / "https_traffic_classifier.joblib")

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
