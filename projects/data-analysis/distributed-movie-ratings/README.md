# Distributed Movie Ratings Analysis

[← Portfolio home](../../../README.md) · [Data Analysis projects](../../../README.md#data-analysis)

> **Publication status:** Source code included.

## Research question

How can Hadoop and Spark identify highly rated movies consistently?

## Dataset

Course ratings.txt, years.txt and genres.txt

## Tools and methods

Python, Hadoop MapReduce and PySpark RDDs

## What I implemented

- Implemented mapper, combiner and reducer components and a Spark pipeline to aggregate ratings and identify top films by genre, with year filtering, a 15-rating threshold and tie handling.
- Documented tests for empty inputs, filters, ties, no-combiner execution and consistency between MapReduce and Spark outputs.

## Results and limitations

Condor comparison is conceptual rather than a demonstrated petabyte-scale run.

## Data access and attribution

Raw datasets, trained model binaries and private operational records are not redistributed. Dataset ownership remains with the respective providers. Where upstream research implementations were adapted, the description distinguishes those methods from local engineering contributions.
