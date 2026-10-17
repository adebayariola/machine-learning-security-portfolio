import json
from pathlib import Path

import numpy as np


EPS = 1e-6


def _to_1d_numeric(flow):
    """Convert one raw flow record into a 1D numeric packet/byte sequence."""
    if isinstance(flow, str):
        parts = flow.replace(",", " ").split()
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                try:
                    values.append(float(int(part, 16)))
                except ValueError:
                    continue
        return np.asarray(values, dtype=np.float32)

    arr = np.asarray(flow)
    if arr.dtype.kind in {"U", "S", "O"}:
        return _to_1d_numeric(" ".join(map(str, arr.reshape(-1).tolist())))

    return arr.astype(np.float32).reshape(-1)


def extract_sequence_features(flow, max_len=1024):
    """
    Create custom numeric features from raw traffic sequences.

    This intentionally does not copy TrafficGPT's GPT/hex/text extraction.
    Channels:
      0. normalized value
      1. sign/direction
      2. log magnitude
      3. normalized absolute difference
      4. cumulative byte/packet mass
      5. non-zero mask
    """
    seq = _to_1d_numeric(flow)
    seq = seq[:max_len]

    if seq.size == 0:
        seq = np.zeros(1, dtype=np.float32)

    abs_seq = np.abs(seq)
    scale = float(np.percentile(abs_seq, 95)) if np.any(abs_seq) else 1.0
    scale = max(scale, EPS)

    norm_value = np.clip(seq / scale, -5.0, 5.0)
    direction = np.sign(seq)
    log_mag = np.log1p(abs_seq) / np.log1p(scale + EPS)
    diff = np.diff(seq, prepend=seq[0])
    diff_scale = max(float(np.percentile(np.abs(diff), 95)), EPS)
    norm_diff = np.clip(diff / diff_scale, -5.0, 5.0)
    cumulative = np.cumsum(abs_seq)
    cumulative = cumulative / max(float(cumulative[-1]), EPS)
    nonzero = (abs_seq > 0).astype(np.float32)

    features = np.stack(
        [norm_value, direction, log_mag, norm_diff, cumulative, nonzero],
        axis=1,
    ).astype(np.float32)

    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    return features


def save_feature_config(output_dir, max_len):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "max_len": max_len,
        "num_channels": 6,
        "channels": [
            "normalized_value",
            "direction_sign",
            "log_magnitude",
            "normalized_difference",
            "cumulative_mass",
            "nonzero_mask",
        ],
    }
    (output_dir / "feature_config.json").write_text(json.dumps(config, indent=2))
