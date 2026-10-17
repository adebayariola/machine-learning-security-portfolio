import argparse
import pickle
from pathlib import Path

import numpy as np

from features import extract_sequence_features, save_feature_config


KNOWN_CLASS_LIMITS = {
    "DF": 60,
    "AWF": 200,
    "DC": 4,
    "USTC": 12,
    "CSTNet": 75,
}


def load_npy_split(dataset_dir):
    x_train = np.load(dataset_dir / "X_train.npy", allow_pickle=True)
    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    x_valid = np.load(dataset_dir / "X_valid.npy", allow_pickle=True)
    y_valid = np.load(dataset_dir / "y_valid.npy", allow_pickle=True)
    x_test = np.load(dataset_dir / "X_test.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_df_split(dataset_dir):
    names = [
        "X_train_NoDef.pkl",
        "y_train_NoDef.pkl",
        "X_valid_NoDef.pkl",
        "y_valid_NoDef.pkl",
        "X_test_NoDef.pkl",
        "y_test_NoDef.pkl",
    ]
    loaded = []
    for name in names:
        with open(dataset_dir / name, "rb") as handle:
            loaded.append(np.asarray(pickle.load(handle, encoding="latin1"), dtype=object))
    return loaded


def deterministic_shuffle(x, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return np.asarray(x, dtype=object)[idx], np.asarray(y)[idx]


def trafficgpt_known_split(x_train, y_train, x_valid, y_valid, x_test, y_test, known_limit, seed):
    """
    Match TrafficGPT's split idea:
      - concatenate original train/test/valid
      - labels below known_limit are known classes
      - labels at/above known_limit are open-set unknowns
      - known data split is 70% train, 18% valid, 12% test
    """
    x_all = np.concatenate([x_train, x_test, x_valid], axis=0)
    y_all = np.concatenate([y_train, y_test, y_valid], axis=0).astype(int)

    known_mask = y_all < known_limit
    x_known = np.asarray(x_all[known_mask], dtype=object)
    y_known = y_all[known_mask]
    x_open = np.asarray(x_all[~known_mask], dtype=object)
    y_open = y_all[~known_mask]

    n = len(y_known)
    train_end = int(n * 0.70)
    valid_end = train_end + int(n * 0.18)

    splits = {
        "train": (x_known[:train_end], y_known[:train_end]),
        "valid": (x_known[train_end:valid_end], y_known[train_end:valid_end]),
        "test": (x_known[valid_end:], y_known[valid_end:]),
        "open": (x_open, y_open),
    }

    for key in ["train", "valid", "test"]:
        splits[key] = deterministic_shuffle(*splits[key], seed=seed)

    return splits


def awf_split(dataset_dir, seed):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_npy_split(dataset_dir)
    open_file = np.load(dataset_dir / "X_open.npz", allow_pickle=True)
    x_open = open_file["data"][:100000]
    y_open = np.full(len(x_open), KNOWN_CLASS_LIMITS["AWF"], dtype=int)
    return {
        "train": deterministic_shuffle(x_train, y_train.astype(int), seed),
        "valid": deterministic_shuffle(x_valid, y_valid.astype(int), seed),
        "test": deterministic_shuffle(x_test, y_test.astype(int), seed),
        "open": (np.asarray(x_open, dtype=object), y_open),
    }


def cstnet_split(dataset_dir):
    splits = {}
    open_x, open_y = [], []
    for split_name in ["train", "valid", "test"]:
        x_items, y_items = [], []
        source = dataset_dir / f"{split_name}.csv"
        if not source.exists():
            source = dataset_dir / f"{split_name}.txt"
        if not source.exists():
            continue
        lines = source.read_text(errors="ignore").splitlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            if "\t" in line:
                label, text = line.split("\t", 1)
            else:
                label, text = line.split(",", 1)
            label = int(label)
            if label < KNOWN_CLASS_LIMITS["CSTNet"]:
                x_items.append(text)
                y_items.append(label)
            else:
                open_x.append(text)
                open_y.append(label)
        splits[split_name] = (np.asarray(x_items, dtype=object), np.asarray(y_items, dtype=int))
    splits["open"] = (np.asarray(open_x, dtype=object), np.asarray(open_y, dtype=int))
    return splits


def load_splits(data_root, dataset, seed):
    dataset = dataset.upper()
    dataset_dir = Path(data_root) / ("CSTNet" if dataset == "CSTNET" else dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    if dataset == "AWF":
        return awf_split(dataset_dir, seed)
    if dataset == "DF":
        return trafficgpt_known_split(*load_df_split(dataset_dir), KNOWN_CLASS_LIMITS["DF"], seed)
    if dataset in {"DC", "USTC"}:
        return trafficgpt_known_split(*load_npy_split(dataset_dir), KNOWN_CLASS_LIMITS[dataset], seed)
    if dataset == "CSTNET":
        return cstnet_split(dataset_dir)

    raise ValueError(f"Unsupported dataset: {dataset}")


def save_split_features(splits, output_dir, max_len):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_feature_config(output_dir, max_len)

    for split_name, (x_values, y_values) in splits.items():
        print(f"Building {split_name} features: {len(y_values)} samples")
        features = np.stack([extract_sequence_features(x, max_len=max_len) for x in x_values])
        labels = np.asarray(y_values, dtype=np.int64)
        np.savez_compressed(output_dir / f"{split_name}.npz", x=features, y=labels)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset using TrafficGPT split rules and custom features.")
    parser.add_argument("--data_root", required=True, help="Folder containing DF/AWF/DC/USTC/CSTNet dataset folders.")
    parser.add_argument("--dataset", required=True, choices=["DF", "AWF", "DC", "USTC", "CSTNet"])
    parser.add_argument("--output_dir", default="processed")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = load_splits(args.data_root, args.dataset, args.seed)
    out = Path(args.output_dir) / args.dataset
    save_split_features(splits, out, args.max_len)
    print(f"Saved processed files to {out}")


if __name__ == "__main__":
    main()
