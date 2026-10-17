# ETGuard Sequence Classification and Alternative Baseline

[← Portfolio home](../../../README.md) · [Security projects](../../../README.md#security)

> **Publication status:** Project summary; source not published.

## Research question

How can packet-length sequences support traffic classification with safer capture-level splits?

## Dataset

USTC-TFC2016 and DoHBrw PCAP adapters

## Tools and methods

Python, Scapy, PyTorch, NumPy and scikit-learn

## What I implemented

- Prepared fixed-length 50-packet sequences for the ETGuard workflow and implemented a file-level USTC split option to reduce overlap between training and test captures.
- Worked with an upstream GRU autoencoder and classifier workflow; added a HistGradientBoosting baseline using packet lengths, distribution summaries and difference statistics, with a configurable decision threshold.
- Prepared model persistence and evaluation outputs. Kept existing result files separate from claims about the safer file-level split because the saved result does not establish which split produced it.

## Results and limitations

Dataset adapters and baseline code are verified; no split-specific score is claimed.

These entries describe inspected implementation work. Published reference scores and unverified draft comparisons are not treated as results of this code. Dataset adapters do not establish completed runs on every supported dataset.

This publication currently contains a project description only. Source files and executable results are not included here.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
