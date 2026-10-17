# Distributed Movie Ratings Analysis

## Data
Course ratings.txt, years.txt and genres.txt

## Tools
Python, Hadoop MapReduce and PySpark RDDs

## Work completed
- Implemented mapper, combiner and reducer components and a Spark pipeline to aggregate ratings and identify top films by genre, with year filtering, a 15-rating threshold and tie handling.
- Documented tests for empty inputs, filters, ties, no-combiner execution and consistency between MapReduce and Spark outputs.

## Evidence and limitations
Condor comparison is conceptual rather than a demonstrated petabyte-scale run.

Raw datasets and private operational records are not distributed.

## Source files
- [mapper.py](distributed-movie-ratings-analysis__mapper.py)
- [reducer.py](distributed-movie-ratings-analysis__reducer.py)
- [combiner.py](distributed-movie-ratings-analysis__combiner.py)
- [spark.py](distributed-movie-ratings-analysis__spark.py)
