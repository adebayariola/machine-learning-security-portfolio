# Open-Set Traffic Classification and CNN State-Mixing Variants

[← Portfolio home](../../../README.md) · [Security projects](../../../README.md#security)

> **Publication status:** Source code included.

## Research question

How can a classifier recognise known traffic and flag traffic from unseen classes?

## Dataset

TrafficGPT-format USTC and ISCX data; configurable DF, AWF, DC and CSTNet preparation

## Tools and methods

Python, PyTorch, NumPy and scikit-learn

## What I implemented

- Adapted TrafficGPT preprocessing for separate CNN and Mamba-style state-mixing training and evaluation without using GPT-2 as the added classifier.
- Implemented a second numeric-sequence variant using direction, magnitude, differences, cumulative traffic mass and masks; added known/unknown class splits and maximum-softmax thresholding.
- Prepared closed-set and open-set classification reports and confusion matrices. Kept upstream reference results separate from results for the added models.

## Results and limitations

Dataset support in code does not establish completed runs on every listed dataset.

These entries describe inspected implementation work. Published reference scores and unverified draft comparisons are not treated as results of this code. Dataset adapters do not establish completed runs on every supported dataset.

## Code and reproduction

[Source files](src/) · [Implementation and command guide](docs/implementation-guide.md)

Run commands from this project directory after installing its dependencies and obtaining the relevant dataset. The implementation guide describes available commands, not evidence that every experiment has been completed.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
