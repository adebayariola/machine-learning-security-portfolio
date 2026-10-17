# Self-Supervised Malware Traffic Classification

[← Portfolio home](../../../README.md) · [Security projects](../../../README.md#security)

> **Publication status:** Project summary; source not published.

## Research question

How can PCAP data be adapted for self-supervised traffic representation learning?

## Dataset

USTC-TFC2016; ISCX VPN-NonVPN

## Tools and methods

Python, PyTorch, Scapy, NumPy and masked autoencoders

## What I implemented

- Adapted the MTC-MAE research implementation with PCAP/archive-to-IDX preparation and one-command runners for pretraining, downstream fine-tuning and testing.
- Added dataset selection, configurable sample limits and short smoke-test settings while retaining the upstream model implementation; supported both Python-script and notebook-style execution.

## Results and limitations

The original model is upstream research code; local adapters and orchestration are the documented implementation work.

These entries describe inspected implementation work. Published reference scores and unverified draft comparisons are not treated as results of this code. Dataset adapters do not establish completed runs on every supported dataset.

This publication currently contains a project description only. Source files and executable results are not included here.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
