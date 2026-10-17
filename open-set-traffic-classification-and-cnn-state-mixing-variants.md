# Open-Set Traffic Classification and CNN State-Mixing Variants

## Data
TrafficGPT-format USTC and ISCX data; configurable DF, AWF, DC and CSTNet preparation

## Tools
Python, PyTorch, NumPy and scikit-learn

## Work completed
- Adapted TrafficGPT preprocessing for separate CNN and Mamba-style state-mixing training and evaluation without using GPT-2 as the added classifier.
- Implemented a second numeric-sequence variant using direction, magnitude, differences, cumulative traffic mass and masks; added known/unknown class splits and maximum-softmax thresholding.
- Prepared closed-set and open-set classification reports and confusion matrices. Kept upstream reference results separate from results for the added models.

## Evidence and limitations
Dataset support in code does not establish completed runs on every listed dataset.

Raw datasets and private operational records are not distributed.

## Source files
- [requirements.txt](open-set-traffic-classification-and-cnn-state-mixing-variants__requirements.txt)
- [prepare_dataset.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__prepare_dataset.py)
- [__init__.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src____init__.py)
- [features.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__features.py)
- [model.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__model.py)
- [dataset.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__dataset.py)
- [train.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__train.py)
- [evaluate.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__evaluate.py)
- [run_all.py](open-set-traffic-classification-and-cnn-state-mixing-variants__src__run_all.py)
