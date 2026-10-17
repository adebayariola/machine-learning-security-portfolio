import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import TrafficFeatureDataset
from model import CNNMambaClassifier


def predict(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def save_confusion_matrix(cm, labels, path, title):
    plt.figure(figsize=(max(7, len(labels) * 0.55), max(5, len(labels) * 0.45)))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN-Mamba with confusion matrices and open-set metrics.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None, help="Max-softmax known threshold. If omitted, use 5th percentile of validation confidence.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    label_map = {int(k): int(v) for k, v in checkpoint["label_map"].items()}
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [str(inv_label_map[i]) for i in range(len(inv_label_map))]

    valid_ds = TrafficFeatureDataset(processed_dir / "valid.npz", label_map=label_map)
    test_ds = TrafficFeatureDataset(processed_dir / "test.npz", label_map=label_map)
    open_ds = TrafficFeatureDataset(processed_dir / "open.npz", label_map=label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMambaClassifier(
        input_channels=checkpoint["input_channels"],
        num_classes=checkpoint["num_classes"],
        hidden_dim=checkpoint["hidden_dim"],
        num_blocks=checkpoint["num_blocks"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    valid_probs, _ = predict(model, valid_ds, args.batch_size, device)
    test_probs, test_y = predict(model, test_ds, args.batch_size, device)

    threshold = args.threshold
    if threshold is None:
        threshold = float(np.percentile(valid_probs.max(axis=1), 5))

    test_pred = test_probs.argmax(axis=1)
    report = classification_report(test_y, test_pred, target_names=class_names, digits=4)
    (output_dir / "closed_set_classification_report.txt").write_text(report)

    cm = confusion_matrix(test_y, test_pred, labels=list(range(len(class_names))))
    save_confusion_matrix(
        cm,
        class_names,
        output_dir / "closed_set_confusion_matrix.png",
        "Closed-Set Confusion Matrix",
    )

    metrics = {
        "closed_set_accuracy": float(accuracy_score(test_y, test_pred)),
        "closed_set_macro_f1": float(f1_score(test_y, test_pred, average="macro")),
        "open_set_threshold": threshold,
    }

    if len(open_ds) > 0:
        open_probs, _ = predict(model, open_ds, args.batch_size, device)
        known_conf = test_probs.max(axis=1)
        unknown_conf = open_probs.max(axis=1)

        known_pred = np.where(known_conf >= threshold, test_probs.argmax(axis=1), len(class_names))
        open_pred = np.where(unknown_conf >= threshold, open_probs.argmax(axis=1), len(class_names))

        combined_true = np.concatenate([test_y, np.full(len(open_pred), len(class_names))])
        combined_pred = np.concatenate([known_pred, open_pred])
        open_labels = class_names + ["unknown"]
        open_cm = confusion_matrix(combined_true, combined_pred, labels=list(range(len(open_labels))))
        save_confusion_matrix(
            open_cm,
            open_labels,
            output_dir / "open_set_confusion_matrix.png",
            "Open-Set Confusion Matrix",
        )

        y_open_binary = np.concatenate([np.zeros(len(known_conf)), np.ones(len(unknown_conf))])
        unknown_score = np.concatenate([1.0 - known_conf, 1.0 - unknown_conf])
        try:
            auroc = float(roc_auc_score(y_open_binary, unknown_score))
        except ValueError:
            auroc = None

        metrics.update(
            {
                "known_accept_rate": float((known_conf >= threshold).mean()),
                "unknown_detection_rate": float((unknown_conf < threshold).mean()),
                "open_set_auroc": auroc,
            }
        )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
