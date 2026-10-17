# CNN-Mamba Open-Set Traffic Classification

This project is prepared to reuse the **same datasets and train/validation/test/open-set split idea** used by TrafficGPT, but it does **not** copy TrafficGPT's GPT-2/hex/text feature extraction.

Our model is:

```text
Raw traffic sequence
-> custom numeric packet/flow feature extraction
-> CNN local sequence encoder
-> Mamba-style long-sequence block
-> classifier
-> open-set detector using max-softmax threshold
```

## What Is Copied From TrafficGPT

Only the dataset preparation logic:

- Dataset names: `DF`, `AWF`, `DC`, `USTC`, `CSTNet`
- Known/open class rule:
  - `DF`: known labels `< 60`, open labels `>= 60`
  - `AWF`: official train/valid/test plus `X_open.npz`
  - `DC`: known labels `< 4`, open labels `>= 4`
  - `USTC`: known labels `< 12`, open labels `>= 12`
  - `CSTNet`: known labels `< 75`, open labels `>= 75`
- For `DF`, `DC`, and `USTC`, known data is split into about:
  - 70% train
  - 18% validation
  - 12% test

## What Is New In This Code

- No GPT-2
- No TabM
- No TrafficGPT text/hex conversion
- Custom numeric sequence features:
  - normalized value
  - packet/byte direction sign
  - log magnitude
  - normalized difference
  - cumulative traffic mass
  - non-zero mask
- CNN-Mamba-style neural network
- Confusion matrix generation
- Closed-set and open-set metrics

## Install

Create a Python environment, then run:

```bash
pip install -r requirements.txt
```

## Dataset Folder Format

Put the downloaded dataset folder somewhere like:

```text
data/
  USTC/
    X_train.npy
    y_train.npy
    X_valid.npy
    y_valid.npy
    X_test.npy
    y_test.npy
```

For `DF`, the folder should contain:

```text
DF/
  X_train_NoDef.pkl
  y_train_NoDef.pkl
  X_valid_NoDef.pkl
  y_valid_NoDef.pkl
  X_test_NoDef.pkl
  y_test_NoDef.pkl
```

For `AWF`, the folder should contain:

```text
AWF/
  X_train.npy
  y_train.npy
  X_valid.npy
  y_valid.npy
  X_test.npy
  y_test.npy
  X_open.npz
```

## One-Command Run

From this folder:

```bash
python src/run_all.py --data_root ./data --dataset USTC --epochs 20 --batch_size 64 --max_len 1024
```

Change `USTC` to one of:

```text
DF
AWF
DC
USTC
CSTNet
```

## Outputs

After running, results are saved in:

```text
outputs/<DATASET>/
```

Important files:

```text
best_cnn_mamba.pt
training_history.json
metrics.json
closed_set_classification_report.txt
closed_set_confusion_matrix.png
open_set_confusion_matrix.png
```

## Run Steps Separately

Prepare features:

```bash
python src/prepare_dataset.py --data_root ./data --dataset USTC --output_dir processed --max_len 1024
```

Train:

```bash
python src/train.py --processed_dir processed/USTC --output_dir outputs/USTC --epochs 20 --batch_size 64
```

Evaluate:

```bash
python src/evaluate.py --processed_dir processed/USTC --model_path outputs/USTC/best_cnn_mamba.pt --output_dir outputs/USTC
```

## Notes For Reporting

You can describe the method like this:

> This work adopts the dataset preparation and open-set split strategy used in TrafficGPT, but replaces GPT-based feature extraction with custom numeric packet/flow sequence features. The proposed classifier uses a CNN encoder to learn local traffic patterns and a Mamba-style state-space sequence block to model long-range dependencies without Transformer attention or TabM.

