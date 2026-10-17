import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate CNN-Mamba traffic model.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", required=True, choices=["DF", "AWF", "DC", "USTC", "CSTNet"])
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "processed" / args.dataset
    output_dir = root / "outputs" / args.dataset

    run([
        sys.executable,
        str(root / "src" / "prepare_dataset.py"),
        "--data_root",
        args.data_root,
        "--dataset",
        args.dataset,
        "--output_dir",
        str(root / "processed"),
        "--max_len",
        str(args.max_len),
        "--seed",
        str(args.seed),
    ])
    run([
        sys.executable,
        str(root / "src" / "train.py"),
        "--processed_dir",
        str(processed_dir),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ])
    run([
        sys.executable,
        str(root / "src" / "evaluate.py"),
        "--processed_dir",
        str(processed_dir),
        "--model_path",
        str(output_dir / "best_cnn_mamba.pt"),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.batch_size),
    ])


if __name__ == "__main__":
    main()
