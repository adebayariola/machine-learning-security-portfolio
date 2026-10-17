from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficFeatureDataset(Dataset):
    def __init__(self, npz_path, label_map=None):
        data = np.load(Path(npz_path), allow_pickle=True)
        self.x = data["x"].astype(np.float32)
        raw_y = data["y"].astype(int)

        if label_map is None:
            classes = sorted(set(raw_y.tolist()))
            label_map = {label: idx for idx, label in enumerate(classes)}
        self.label_map = label_map
        self.y = np.asarray([label_map.get(int(label), -1) for label in raw_y], dtype=np.int64)
        keep = self.y >= 0
        self.x = self.x[keep]
        self.y = self.y[keep]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def build_label_map(train_npz):
    y = np.load(Path(train_npz), allow_pickle=True)["y"].astype(int)
    classes = sorted(set(y.tolist()))
    return {label: idx for idx, label in enumerate(classes)}
