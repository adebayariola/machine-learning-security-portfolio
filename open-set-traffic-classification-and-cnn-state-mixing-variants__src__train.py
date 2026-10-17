import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrafficFeatureDataset, build_label_map
from model import CNNMambaClassifier


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(mode=train)
    losses, preds, labels = [], [], []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN-Mamba traffic classifier.")
    parser.add_argument("--processed_dir", required=True, help="Processed dataset folder, e.g. processed/USTC")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = build_label_map(processed_dir / "train.npz")
    train_ds = TrafficFeatureDataset(processed_dir / "train.npz", label_map=label_map)
    valid_ds = TrafficFeatureDataset(processed_dir / "valid.npz", label_map=label_map)

    input_channels = train_ds.x.shape[-1]
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNMambaClassifier(
        input_channels=input_channels,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_f1 = -1.0
    best_path = output_dir / "best_cnn_mamba.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_f1={valid_metrics['macro_f1']:.4f}"
        )

        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_map": label_map,
                    "input_channels": input_channels,
                    "num_classes": num_classes,
                    "hidden_dim": args.hidden_dim,
                    "num_blocks": args.num_blocks,
                    "dropout": args.dropout,
                },
                best_path,
            )

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved best model to {best_path}")


if __name__ == "__main__":
    main()
