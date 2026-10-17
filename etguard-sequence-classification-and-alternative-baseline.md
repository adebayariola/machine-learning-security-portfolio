# ETGuard Sequence Classification and Alternative Baseline

## Data
USTC-TFC2016 and DoHBrw PCAP adapters

## Tools
Python, Scapy, PyTorch, NumPy and scikit-learn

## Work completed
- Prepared fixed-length 50-packet sequences for the ETGuard workflow and implemented a file-level USTC split option to reduce overlap between training and test captures.
- Worked with an upstream GRU autoencoder and classifier workflow; added a HistGradientBoosting baseline using packet lengths, distribution summaries and difference statistics, with a configurable decision threshold.
- Prepared model persistence and evaluation outputs. Kept existing result files separate from claims about the safer file-level split because the saved result does not establish which split produced it.

## Evidence and limitations
Dataset adapters and baseline code are verified; no split-specific score is claimed.

Raw datasets and private operational records are not distributed.

This entry documents the project; source code is not included in this publication.
